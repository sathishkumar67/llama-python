from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


@dataclass
class DataArgs:
    batch_size: int
    block_size: int
    pad_token_id: int = 0


class TokenDataset(Dataset):
    def __init__(self, input_ids, args: DataArgs):
        self.input_ids = input_ids
        self.block_size = args.block_size
        self.pad_token_id = args.pad_token_id

    def __len__(self):
        # Number of full blocks
        return (len(self.input_ids) + self.block_size - 1) // self.block_size

    def __getitem__(self, idx):
        start_idx = idx * self.block_size
        end_idx = start_idx + self.block_size
        input_ids_block = self.input_ids[start_idx:end_idx]

        # If the block is shorter than block_size, pad it
        if len(input_ids_block) < self.block_size:
            padding_length = self.block_size - len(input_ids_block)
            input_ids_block += [self.pad_token_id] * padding_length

        return torch.tensor(input_ids_block)
    


def collate_fn(batch):
    """
    A function that dynamically pads the sequences in the batch.

    Args:
        batch: The input batch of sequences to pad.

    Returns:
        padded_batch: The padded batch of sequences.

    Note:
        The padding value used is 0, which ensures that the padded tokens do not have interaction with other useful tokens.
    """
    # Dynamically pad the sequences in the batch
    padded_batch = pad_sequence(batch, batch_first=True, padding_value=0)
    return padded_batch


