import torch
from torch.utils.data import Dataset

class NextTokenDataset(Dataset):
    def __init__(self, tokenized_texts, seq_len=10):
        self.samples = []  # список кортежей (X, Y)
        for tokens in tokenized_texts:
            if len(tokens) <= seq_len:
                continue
            # Скользящее окно X = первые seq_len токенов, Y = следующий токен
            for i in range(len(tokens) - seq_len):
                x = tokens[i:i+seq_len]
                y = tokens[i+seq_len]
                self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)