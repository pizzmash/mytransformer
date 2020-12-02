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

