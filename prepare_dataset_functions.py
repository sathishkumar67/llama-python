def split_dataframe(df, num_splits):
    # Determine the length of the DataFrame
    n = len(df)
    
    # Calculate the indices for the splits
    indices = [i * (n // num_splits) for i in range(1, num_splits)]
    
    # Split the DataFrame
    splits = []
    previous_index = 0
    for index in indices:
        splits.append(df.iloc[previous_index:index])
        previous_index = index
    splits.append(df.iloc[previous_index:])
    
    return splits

# convert the dataframe to tokens
def convert_to_tokens(splitted_df, tokenizer, cols):
    bagoftokens = []
    for _, df in enumerate(splitted_df):
        if len(cols) == 1:
            df["text"] = df[cols[0]]
        else:
            df["text"] = df[cols[0]]
            for col in cols[1:]:
                df["text"] += ". " + df[col]
        
        text = ". ".join(df["text"])
        tokens = tokenizer(text).input_ids
        bagoftokens.append(tokens)
    return bagoftokens

def convert_to_binary(bag_of_tokens, directory_name, file_name):
    # create a directory to store binary files
    os.makedirs(directory_name, exist_ok=True)
    for i, tokens in enumerate(bag_of_tokens):
        # File path to store the bytestring
        file_path = f'{directory_name}/{file_name}_{i}.bin'

        # Store the bytestring in a file
        with open(file_path, 'wb') as file:
            file.write(pickle.dumps(tokens))
