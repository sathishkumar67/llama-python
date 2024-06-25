class PrepareDataset:
    def __init__(self, df, tokenizer, cols, num_splits, directory_name, file_name):
        self.df = df
        self.tokenizer = tokenizer
        self.cols = cols
        self.num_splits = num_splits
        self.directory_name = directory_name
        self.file_name = file_name
        self.bag_of_tokens = []

    def split_dataframe(self):
        n = len(self.df)
        indices = [i * (n // self.num_splits) for i in range(1, self.num_splits)]
        splits = []
        previous_index = 0
        for index in indices:
            splits.append(self.df.iloc[previous_index:index])
            previous_index = index
        splits.append(self.df.iloc[previous_index:])
        return splits

    def convert_to_tokens(self, splitted_df):
        bagoftokens = []
        for _, df in enumerate(splitted_df):
            if len(self.cols) == 1:
                df["text"] = df[self.cols[0]]
            else:
                df["text"] = df[self.cols[0]]
                for col in self.cols[1:]:
                    df["text"] += ". " + df[col]
            text = ". ".join(df["text"])
            tokens = self.tokenizer(text).input_ids
            bagoftokens.append(tokens)
        self.bag_of_tokens = bagoftokens

    def convert_to_binary(self):
        os.makedirs(self.directory_name, exist_ok=True)
        for i, tokens in enumerate(self.bag_of_tokens):
            file_path = f'{self.directory_name}/{self.file_name}_{i}.bin'
            with open(file_path, 'wb') as file:
                file.write(pickle.dumps(tokens))

    def execute(self):
        splitted_df = self.split_dataframe()
        self.convert_to_tokens(splitted_df)
        self.convert_to_binary()
