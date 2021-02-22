import pickle
import sentencepiece as spm
import torch
from torch.utils.data import Dataset
from torch.nn.utils import rnn

class SentencePieceDataset(Dataset):
    def __init__(self, xmodel_path, ymodel_path, data_path, max_xlen, max_ylen):
        self.spx = spm.SentencePieceProcessor()
        self.spx.load(xmodel_path)
        self.spy = spm.SentencePieceProcessor()
        self.spy.load(ymodel_path)
        with open(data_path, mode='rb') as f:
            data = pickle.load(f)
        self.x = [torch.tensor(d[0][:max_xlen]) for d in data]
        self.y = [torch.tensor(d[1][:max_ylen]) for d in data]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

    def collate_fn(self, batch):
        x = [d[0] for d in batch]
        x = rnn.pad_sequence(x, batch_first=True, padding_value=self.spx['<pad>'])
        y = [d[1] for d in batch]
        y = rnn.pad_sequence(y, batch_first=True, padding_value=self.spy['<pad>'])
        return x, y


class MyDataset(Dataset):
    def __init__(self, xmodel_path, ymodel_path, data_path, max_xlen, max_ylen):
        self.spx = spm.SentencePieceProcessor()
        self.spx.load(xmodel_path)
        self.spy = spm.SentencePieceProcessor()
        self.spy.load(ymodel_path)
        with open(data_path, mode='rb') as f:
            data = pickle.load(f)
        self.x = [torch.tensor(d[0][:max_xlen]) for d in data]
        self.y = [torch.tensor(d[1][:max_ylen]) for d in data]
        # なんか変！
        # rankss = [d[2][:max_xlen] for d in data]
        # rank_valuess = [list(set(ranks)) for ranks in rankss]
        # ths = [3 if len(rank_values) >= 4 else len(rank_values) for rank_values in rank_valuess]
        # self.z = [torch.tensor([1 if 0 <= rank <= sorted(rank_values)[th] else 0 for rank in ranks]) for ranks, rank_values, th in zip(rankss, rank_valuess, ths)]
        self.z = [torch.zeros_like(x) for x in self.x]

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.z[index]

    def __len__(self):
        return len(self.x)

    def collate_fn(self, batch):
        x = [d[0] for d in batch]
        x = rnn.pad_sequence(x, batch_first=True, padding_value=self.spx['<pad>'])
        y = [d[1] for d in batch]
        y = rnn.pad_sequence(y, batch_first=True, padding_value=self.spy['<pad>'])
        z = [d[2] for d in batch]
        z = rnn.pad_sequence(z, batch_first=True, padding_value=0)
        return x, y, z
    
