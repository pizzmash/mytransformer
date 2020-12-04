from dataloader.dataset import SentencePieceDataset
from model.transformer import Transformer
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F


# train_ds = SentencePieceDataset('../data/sp/article.model',
#                                 '../data/sp/summary.model',
#                                 '../data/ds/train.pickle',
#                                 400, 100)
# valid_ds = SentencePieceDataset('../data/sp/article.model',
#                                 '../data/sp/summary.model',
#                                 '../data/ds/valid.pickle',
#                                 400, 100)
test_ds = SentencePieceDataset('../data/sp/article.model',
                               '../data/sp/summary.model',
                               '../data/ds/test.pickle',
                               400, 10000)

# train_dl = DataLoader(train_ds,
#                       batch_size=16,
#                       shuffle=True,
#                       collate_fn=train_ds.collate_fn)
# valid_dl = DataLoader(valid_ds,
#                       batch_size=16,
#                       shuffle=False,
#                       collate_fn=valid_ds.collate_fn)


model = Transformer(source_vocab_length=len(test_ds.spx),
                    target_vocab_length=len(test_ds.spy))

optim = torch.optim.Adam(model.parameters(), lr=0.0001,
                         betas=(0.9, 0.98), eps=1e-9)


def greedy_decode_sentence(model, ids):
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model.eval()
  indexed = ids
  sentence = torch.autograd.Variable(torch.LongTensor([indexed])).to(device)
  tgt = torch.LongTensor([[test_ds.spy['<s>']]]).to(device)
  translated_ids = []
  maxlen = 100
  for i in range(maxlen):
    size = tgt.size(0)
    np_mask = torch.triu(torch.ones(size, size)==1).transpose(0,1)
    np_mask = np_mask.float().masked_fill(np_mask == 0, float('-inf')).masked_fill(np_mask == 1, float(0.0))
    np_mask = np_mask.to(device)
    pred = model(sentence.transpose(0,1), tgt, tgt_mask=np_mask)
    add_id = int(pred.argmax(dim=2)[-1])
    translated_ids.append(add_id)
    if add_id == test_ds.spy['</s>']:
      break
    tgt = torch.cat((tgt, torch.LongTensor([[add_id]]).to(device)))
  # return test_ds.spy.DecodeIds(translated_ids)
  return translated_ids


def train(train_iter, val_iter, model, optim, num_epochs, use_gpu=True):
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model.to(device)
  torch.backends.cudnn.benchmark = True

  dataloaders_dict = {'train': train_iter, 'val': val_iter}
  losses_dict = {'train': [], 'val': []}

  for epoch in range(num_epochs):
    for phase in ['train', 'val']:
      if phase == 'train':
        model.train()
      else:
        model.eval()

      epoch_loss = 0.0

      for i, batch in enumerate(dataloaders_dict[phase]):
        src = batch[0].to(device)
        tgt = batch[1].to(device)

        optim.zero_grad()
        with torch.set_grad_enabled(phase == 'train'):
          tgt_input = tgt[:, :-1]
          tgt_output = tgt[:, 1:].contiguous().view(-1)

          src_mask = (src != train_ds.spx['<pad>'])
          src_mask = src_mask.float().masked_fill(src_mask == 0, float('-inf'))
          src_mask = src_mask.masked_fill(src_mask == 1, float(0.0))
          src_mask = src_mask.to(device)

          tgt_mask = (tgt_input != train_ds.spy['<pad>'])
          tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf'))
          tgt_mask = tgt_mask.masked_fill(tgt_mask == 1, float(0.0))
          tgt_mask = tgt_mask.to(device)

          size = tgt_input.size(1)
          np_mask = torch.triu(torch.ones(size, size) == 1).transpose(0, 1)
          np_mask = np_mask.float().masked_fill(np_mask == 0, float('-inf'))
          np_mask = np_mask.masked_fill(np_mask == 1, float(0.0))
          np_mask = np_mask.to(device)

          preds = model(src.transpose(0, 1), tgt_input.transpose(0, 1), tgt_mask=np_mask)
          preds = preds.transpose(0,1).contiguous().view(-1, preds.size(-1))
          loss = F.cross_entropy(preds, tgt_output, ignore_index=train_ds.spy['<pad>'], reduction='sum')

          if phase == 'train':
            loss.backward()
            optim.step()

          batch_loss = loss.item() / 16
          epoch_loss += batch_loss

          if phase == 'train' and i % 10 == 0:
            print('Epoch {}/{} | Batch {}/{} | {:^5} | Loss: {:.4f}'.format(epoch+1,
                                                                            num_epochs,
                                                                            i+1,
                                                                            len(dataloaders_dict[phase]),
                                                                            phase,
                                                                            batch_loss))
            if i % 100 == 0:
              print(greedy_decode_sentence(model, valid_ds[0][0].tolist()))

      epoch_loss = epoch_loss / len(dataloaders_dict[phase])
      print('Epoch {}/{} | {:^5} | Loss: {:.4f}'.format(epoch+1,
                                                        num_epochs,
                                                        phase,
                                                        epoch_loss))

      if phase == 'val':
        if epoch_loss < min(losses_dict['val'], default=1e9):
          print("saving state dict")
          torch.save(model.state_dict(), f"checkpoint_best_epoch.pt")

      losses_dict[phase].append(epoch_loss)

  return losses_dict['train'], losses_dict['val']       


def split_and_decode(ids):
  decoded = []
  while True:
    if test_ds.spy['<sep>'] not in ids:
      decoded.append(test_ds.spy.DecodeIds(ids))
      return decoded
    else:
      idx = ids.index(test_ds.spy['<sep>'])
      decoded.append(test_ds.spy.DecodeIds(ids[:idx]))
      if len(ids) - 1 == idx:
        return decoded
      else:
        ids = ids[idx+1:]


def save_summary(path, summaries):
  with open(path, mode="w") as f:
    f.write('\n'.join(summaries))


def test(model):
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model.to(device)
  model.load_state_dict(torch.load('checkpoint_best_epoch.pt'))

  torch.backends.cudnn.benchmark = True

  reference_dir = "test/reference/"
  decoded_dir = "test/decoded/"

  for i, (article, summary) in enumerate(zip(test_ds.x, test_ds.y)):
    reference_summaries = split_and_decode(summary.tolist())
    save_summary(reference_dir + str(i).zfill(5) + ".txt", reference_summaries)
    decoded = greedy_decode_sentence(model, article.tolist())
    decoded_summaries = split_and_decode(decoded)
    save_summary(decoded_dir + str(i).zfill(5) + ".txt", decoded_summaries)
    
    

# train_losses, valid_losses = train(train_dl, valid_dl, model, optim, 10)
test(model)


