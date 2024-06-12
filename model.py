import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Optional


# default arguments
@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 20
    n_heads: int = 16
    n_kv_heads: Optional[int] = 8 # This should be half the n_heads
    vocab_size: int = 32000 # Later set in the build method
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    # Needed for KV cache
    max_batch_size: int = 8
    max_seq_len: int = 512

    device: str = "cpu" # Later we 
    
    is_causal: bool = True


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):

        """
        Initializes the RMSNorm layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): The epsilon value for numerical stability. Defaults to 1e-6.

        Returns:
            None
        """

        super().__init__()
        self.eps = eps
        # The gamma parameter
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        
        """
        Normalizes the input tensor `x` using the RMSNorm technique.

        Args:
            x (torch.Tensor): The input tensor to be normalized.

        Returns:
            torch.Tensor: The normalized tensor.

        The function calculates the square root of the inverse of the mean of the squared values of `x` along the last dimension. It then adds the epsilon value for numerical stability. The result is multiplied by `x` to obtain the normalized tensor.

        Note:
            The `keepdim` argument is set to `True` to ensure that the output tensor has the same shape as the input tensor, except for the last dimension which is reduced to 1.
        """
        # rsqrt: 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        
        """
        Calculates the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor of shape (B, Seq_Len, Dim).

        Returns:
            torch.Tensor: The output tensor of shape (B, Seq_Len, Dim).

        The function takes in a tensor `x` of shape (B, Seq_Len, Dim) and performs the following operations:
        
        - It applies the `_norm` function to `x` to normalize the values.
        - It multiplies the result of `_norm` with the `weight` parameter.
        - It casts the result to the same type as `x`.

        The final output tensor is of shape (B, Seq_Len, Dim).

        Note:
            The `_norm` function is not defined in the code snippet.
        """       
        # (Dim) * (B, Seq_Len, Dim) = (B, Seq_Len, Dim)
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):

    """
    Precomputes the theta position frequencies for a given head dimension, sequence length, and device.

    Args:
        head_dim (int): The head dimension.
        seq_len (int): The sequence length.
        device (str): The device to compute the frequencies on.
        theta (float, optional): The theta parameter. Defaults to 10000.0.

    Returns:
        torch.Tensor: The complex frequencies tensor of shape (seq_len, head_dim // 2).

    Raises:
        AssertionError: If the head dimension is not divisible by 2.

    This function precomputes the theta position frequencies for a given head dimension, sequence length, and device.
    It asserts that the head dimension is divisible by 2. It builds the theta parameter using the formula theta_i =
    10000^(-2(i-1)/dim) for i = [1, 2, ... dim/2]. It constructs the positions (the "m" parameter) using the arange
    function. It then multiplies each theta by each position using the outer product. Finally, it computes complex
    numbers in the polar form c = R * exp(m * theta) using the torch.polar function.

    Note:
        The theta parameter is a float and defaults to 10000.0.
    """
    # As written in the paragraph 3.2.2 of the paper
    # >> In order to generalize our results in 2D to any xi ∈ Rd where **d is even**, [...]
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"
    # Build the theta parameter
    # According to the formula theta_i = 10000^(-2(i-1)/dim) for i = [1, 2, ... dim/2]
    # Shape: (Head_Dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    # Shape: (Head_Dim / 2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device) # (Dim / 2)
    # Construct the positions (the "m" parameter)
    # Shape: (Seq_Len)
    m = torch.arange(seq_len, device=device)
    # Multiply each theta by each position using the outer product.
    # Shape: (Seq_Len) outer_product* (Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
    freqs = torch.outer(m, theta).float()
    # We can compute complex numbers in the polar form c = R * exp(m * theta), where R = 1 as follows:
    # (Seq_Len, Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)

    return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):

    """
    Apply rotary embeddings to the input tensor.

    Args:
        x (torch.Tensor): The input tensor of shape (B, Seq_Len, H, Head_Dim).
        freqs_complex (torch.Tensor): The complex frequencies tensor of shape (Seq_Len, Head_Dim/2).
        device (str): The device to use for the computation.

    Returns:
        torch.Tensor: The output tensor of shape (B, Seq_Len, H, Head_Dim) after applying rotary embeddings.

    This function separates the last dimension pairs of two values, representing the real and imaginary parts of the complex number.
    Two consecutive values will become a single complex number. The input tensor is reshaped to (B, Seq_Len, H, Head_Dim/2).
    The freqs_complex tensor is reshaped to match the shape of the x_complex tensor, adding the batch dimension and the head dimension.
    The complex numbers in the x_complex tensor are multiplied by the corresponding complex numbers in the freqs_complex tensor,
    resulting in the rotation of the complex number as shown in the Figure 1 of the paper. The output tensor is reshaped back to
    (B, Seq_Len, H, Head_Dim) and returned.
    """
    # Separate the last dimension pairs of two values, representing the real and imaginary parts of the complex number
    # Two consecutive values will become a single complex number
    # (B, Seq_Len, H, Head_Dim) -> (B, Seq_Len, H, Head_Dim/2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # Reshape the freqs_complex tensor to match the shape of the x_complex tensor. So we need to add the batch dimension and the head dimension
    # (Seq_Len, Head_Dim/2) --> (1, Seq_Len, 1, Head_Dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # Multiply each complex number in the x_complex tensor by the corresponding complex number in the freqs_complex tensor
    # Which results in the rotation of the complex number as shown in the Figure 1 of the paper
    # (B, Seq_Len, H, Head_Dim/2) * (1, Seq_Len, 1, Head_Dim/2) = (B, Seq_Len, H, Head_Dim/2)
    x_rotated = x_complex * freqs_complex
    # Convert the complex number back to the real number
    # (B, Seq_Len, H, Head_Dim/2) -> (B, Seq_Len, H, Head_Dim/2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (B, Seq_Len, H, Head_Dim/2, 2) -> (B, Seq_Len, H, Head_Dim)
    x_out = x_out.reshape(*x.shape)

    return x_out.type_as(x).to(device)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:

    """
    Repeats the key-value heads of a tensor.

    Args:
        x (torch.Tensor): The input tensor of shape (B, Seq_Len, N_KV_Heads, Head_Dim).
        n_rep (int): The number of times to repeat the key-value heads.

    Returns:
        torch.Tensor: The output tensor of shape (B, Seq_Len, N_KV_Heads * N_Rep, Head_Dim) after repeating the key-value heads.

    This function repeats the key-value heads of a tensor by expanding the tensor along the key-value dimension and then reshaping it. If n_rep is 1, the input tensor is returned as is.
    """

    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        # (B, Seq_Len, N_KV_Heads, 1, Head_Dim)
        x[:, :, :, None, :]
        # (B, Seq_Len, N_KV_Heads, N_Rep, Head_Dim)
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        # (B, Seq_Len, N_KV_Heads * N_Rep, Head_Dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )


class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):

        """
        Initializes the class with the given arguments.

        Args:
            args (ModelArgs): The arguments for the model.

        Returns:
            None

        This method initializes the class with the given arguments. It sets the arguments of the model, the number of heads for the Keys and Values, the number of heads for the Queries, the number of times the Keys and Values should be repeated, and the dimension of each head. It also initializes the weights for query, key, value, and output. Finally, it precomputes the theta position frequencies.

        """
        super().__init__()

        # arguments of the model
        self.args = args

        # Indicates the number of heads for the Keys and Values
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # Indicates the number of heads for the Queries
        self.n_heads_q = args.n_heads
        # Indicates how many times the Keys and Values should be repeated
        self.n_rep = self.n_heads_q // self.n_kv_heads
        # Indicates the dimension of each head, that is, the part of the embedding that each head will be responsible for
        self.head_dim = args.dim // args.n_heads

        # Weights for query, key, value and output
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len, device=self.args.device)

    def forward(
        self,
        x: torch.Tensor
    ):
        
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor of shape (B, Seq_Len, Dim).

        Returns:
            torch.Tensor: The output tensor of shape (B, Seq_Len, Dim).

        This method performs the forward pass of the model. It takes in a tensor `x` of shape (B, Seq_Len, Dim) and applies various transformations to it.

        The method first extracts the batch size, sequence length, and dimension of the input tensor.

        Then, it applies linear transformations to the input tensor using the `wq`, `wk`, and `wv` weights.

        The resulting tensors are reshaped to have shape (B, Seq_Len, N_Heads_Q, Head_Dim), (B, Seq_Len, N_KV_Heads, Head_Dim), and (B, Seq_Len, N_KV_Heads, Head_Dim), respectively.

        The query tensor is transformed using the `apply_rotary_embeddings` function, which applies rotary embeddings to the tensor.

        The key and value tensors are repeated using the `repeat_kv` function, which repeats the key and value heads for every query in the same group.

        The tensors are transposed and the attention scores are calculated by taking the dot product of the query and key tensors and dividing by the square root of the head dimension.

        If the model is causal, a causal attention mask is applied to the attention scores.

        The attention scores are then softmaxed and transformed to the same type as the query tensor.

        The output is calculated by taking the dot product of the attention scores and the value tensor."""
        
        # (B, Seq_Len, Dim) -> x.shape
        batch_size, seq_len, _ = x.shape  

        # (B, Seq_Len, Dim) -> (B, Seq_Len, Dim)
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)


        xq = apply_rotary_embeddings(xq, self.freqs_complex, device=x.device)
        xk = apply_rotary_embeddings(xk, self.freqs_complex, device=x.device)

        # Since every group of Q shares the same K and V heads, just repeat the K and V heads for every Q in the same group.
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # causal attention calculation
        attn_scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)

        # causal attention mask
        if self.args.is_causal:
            causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)
            attn_scores.masked_fill(causal_mask == 0, -float("inf"))

        attn_scores = F.softmax(attn_scores.float(), dim=-1).type_as(xq)

        output = torch.matmul(attn_scores, xv)
        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))

        return self.wo(output) 


class FeedForward(nn.Module):
    def __init__(
        self,
        args: ModelArgs
    ):
        
        """
        Initializes the FeedForward class with the given arguments.

        Args:
            args (ModelArgs): The arguments for the model.

        Returns:
            None

        This method initializes the FeedForward class with the given arguments. It calculates the hidden_dim by multiplying args.dim by 4 and then dividing by 3. If args.ffn_dim_multiplier is not None, it multiplies hidden_dim by args.ffn_dim_multiplier. The hidden_dim is then rounded to the nearest multiple of args.multiple_of. The class initializes three linear layers with input dimension args.dim, output dimension hidden_dim, and no bias.
        """

        super().__init__()

        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        # Round the hidden_dim to the nearest multiple of the multiple_of parameter
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        """
        Apply the forward pass of the model to the input tensor `x`.

        Args:
            x (torch.Tensor): The input tensor of shape (B, Seq_Len, Dim).

        Returns:
            torch.Tensor: The output tensor of shape (B, Seq_Len, Dim) after applying the forward pass.

        The function applies the forward pass of the model to the input tensor `x`. It performs the following operations:

        1. It applies the `silu` activation function to the output of the linear layer `self.w1` to obtain the tensor `swish`.
        2. It applies the linear layer `self.w3` to the input tensor `x` to obtain the tensor `x_V`.
        3. It multiplies the tensor `swish` with the tensor `x_V` element-wise to obtain the tensor `x`.
        4. It applies the linear layer `self.w2` to the tensor `x` to obtain the final output tensor.

        The output tensor has the same shape as the input tensor.
        """

        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        swish = F.silu(self.w1(x))
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        x_V = self.w3(x)
        # (B, Seq_Len, Hidden_Dim) * (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Hidden_Dim)
        x = swish * x_V
        # (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Dim)
        x = self.w2(x)
        return x


class EncoderBlock(nn.Module):

    def __init__(self, args: ModelArgs):
        """
        Initializes an instance of the EncoderBlock class.

        Args:
            args (ModelArgs): The arguments for the model.

        Returns:
            None

        This method initializes an instance of the EncoderBlock class with the given arguments. It sets the number of heads, the dimension, and the head dimension based on the arguments. It also initializes the attention and feed forward layers. Additionally, it initializes the attention_norm and ffn_norm normalization layers with the specified epsilon value.

        """
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        # Normalization BEFORE the attention block
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # Normalization BEFORE the feed forward block
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor of shape (B, Seq_Len, Dim).

        Returns:
            torch.Tensor: The output tensor of shape (B, Seq_Len, Dim) after applying the forward pass.

        This method performs the forward pass of the model. It takes in a tensor `x` of shape (B, Seq_Len, Dim) and applies the following operations:

        1. It adds the output of the attention layer to the input tensor `x` to obtain the tensor `h`.
        2. It adds the output of the feed forward layer to the tensor `h` to obtain the final output tensor.

        The final output tensor has the same shape as the input tensor.
        """
        # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) --> (B, Seq_Len, Dim)
        h = x + self.attention.forward(self.attention_norm(x))
        # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) --> (B, Seq_Len, Dim)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out
    
class Transformer(nn.Module):

    def __init__(self, args: ModelArgs):
        """
        Initializes the Transformer class.

        Args:
            args (ModelArgs): The arguments for the model.

        Returns:
            None

        This method initializes the Transformer class with the given arguments. It sets the vocab_size, n_layers, and dim attributes based on the arguments. It also initializes the tok_embeddings, layers, norm, and output attributes. The tok_embeddings is an nn.Embedding layer with the vocab_size and dim from the arguments. The layers is a nn.ModuleList containing EncoderBlock instances with the arguments. The norm is an RMSNorm layer with the dim and eps from the arguments. The output is an nn.Linear layer with the dim and vocab_size from the arguments.

        """
        super().__init__()

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

    def activate(self):
        """
        Activates the Transformer model by creating a sequential model with the token embeddings, the layers, the norm layer, and the output layer.

        Returns:
            nn.Sequential: The sequential model containing the token embeddings, the layers, the norm layer, and the output layer.
        """
        return nn.Sequential(self.tok_embeddings, 
                             *self.layers, 
                             self.norm, 
                             self.output)

# # args for trainer class
# @dataclass
# class TrainerArgs:
#     model: nn.Module
#     optimizer: torch.optim.Optimizer
#     loss_fn: nn.modules.loss._Loss

#     epochs: int
#     learning_rate: int
#     batch_size: int
#     block_size: int



# # trainer class for the model
# class Trainer():
#     def __init__(self, model: nn.Module | nn.Sequential, optimizer: torch.optim.Optimizer, loss_fn: nn.modules.loss._Loss):
#         """
#         Initializes the Trainer class with the given model, optimizer, and loss function.

#         Args:
#             model (nn.Module): The model to be trained.
#             optimizer (torch.optim.Optimizer): The optimizer to be used for training.
#             loss_fn (nn.modules.loss._Loss): The loss function to be used for training.

#         Returns:
#             None
#         """
#         self.model = model
#         self.optimizer = optimizer
#         self.loss_fn = loss_fn

    