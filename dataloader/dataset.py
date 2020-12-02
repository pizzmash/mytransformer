import pickle
import sentencepiece as spm
from torch.utils.data import Dataset


class SentencePieceDataset(Dataset):
  def __init__(self, model_path, data_path):
    self.sp = spm.SentencePieceProcessor()
    self.sp.load(model_path)
    with open(data_path, mode='rb') as f:
      self.data = pickle.load(f)

  def __getitem__(self, index)
    return self.data[index][0], self.data[index][1]

  def __len__(self):
    return len(self.data)
