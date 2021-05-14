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
        self.x = [torch.tensor(d[0][:max_xlen]) for d in data if len(d[0]) > 0]
        self.y = [torch.tensor(d[1][:max_ylen]) for d in data if len(d[0]) > 0]

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
    def __init__(self, xmodel_path, ymodel_path, data_path, max_xlen, max_ylen, bin_imp=True):
        self.spx = spm.SentencePieceProcessor()
        self.spx.load(xmodel_path)
        self.spy = spm.SentencePieceProcessor()
        self.spy.load(ymodel_path)
        with open(data_path, mode='rb') as f:
            data = pickle.load(f)
        self.x = [torch.tensor(d[0][:max_xlen]) for d in data if len(d[0]) > 0]
        self.y = [torch.tensor(d[1][:max_ylen]) for d in data if len(d[0]) > 0]
        if bin_imp:
          # 各データの単語対応位置の文の重要度ランク
          ranks_list = [d[2][:max_xlen] for d in data if len(d[0]) > 0]
          # 各データの文の数
          n_sentences_list = [max(ranks) + 1 for ranks in ranks_list]
          # 各データに対して何番目のランクの文まで重要とするか
          ths = [3 if n_sentences > 3 else n_sentences  for n_sentences in n_sentences_list]
          # 各データの単語対応位置の文が重要かどうか
          self.z = [torch.tensor([1 if 0 <= rank <= th else 0 for rank in ranks])
                    for ranks, n_sentences, th in zip(ranks_list, n_sentences_list, ths)]
        else:
          self.z = [torch.tensor(d[2][:max_xlen]) for d in data if len(d[0]) > 0]

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
    
