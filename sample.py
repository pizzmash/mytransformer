from dataloader.dataset import SentencePieceDataset
from torch.utils.data import DataLoader


ds = SentencePieceDataset('../data/sp/article.model',
                          '../data/sp/summary.model',
                          '../data/ds/test.pickle',
                          400, 100)
dl = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=ds.collate_fn)


