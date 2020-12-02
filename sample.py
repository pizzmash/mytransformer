from dataloader.dataset import SentencePieceDataset
from model.transformer import Transformer
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F


train_ds = SentencePieceDataset('../data/sp/article.model',
                                '../data/sp/summary.model',
                                '../data/ds/train.pickle',
                                400, 100)
valid_ds = SentencePieceDataset('../data/sp/article.model',
                                '../data/sp/summary.model',
                                '../data/ds/valid.pickle',
                                400, 100)
train_dl = DataLoader(train_ds,
                      batch_size=16,
                      shuffle=True,
                      collate_fn=train_ds.collate_fn)
valid_dl = DataLoader(valid_ds,
                      batch_size=16,
                      shuffle=False,
                      collate_fn=train_ds.collate_fn)


model = Transformer(source_vocab_length=len(train_ds.spx),
                    target_vocab_length=len(train_ds.spy))

optim = torch.optim.Adam(model.parameters(), lr=0.0001,
                         betas=(0.9, 0.98), eps=1e-9)


def greedy_decode_sentence(model, ids):
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model.eval()
  indexed = ids
  sentence = torch.autograd.Variable(torch.LongTensor([indexed])).to(device)
  tgt = torch.LongTensor([[train_ds.spy['<s>']]]).to(device)
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
    if add_id == train_ds.spy['</s>']:
      break
    tgt = torch.cat((tgt, torch.LongTensor([[add_id]]).to(device)))
  return train_ds.spy.DecodeIds(translated_ids)


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

  """
  train_losses = []
  valid_losses = []
  for epoch in range(num_epochs):
    train_loss = 0
    valid_loss = 0
    model.train()
    for i, batch in enumerate(train_iter):
      src = batch[0].to(device)
      tgt = batch[1].to(device)
      tgt_input = tgt[:, :-1]
      targets = tgt[:, 1:].contiguous().view(-1)
      src_mask = (src != train_ds.spx['<pad>'])
      src_mask = src_mask.float().masked_fill(src_mask == 0, float('-inf')).masked_fill(src_mask == 1, float(0.0))
      src_mask = src_mask.to(device)
      tgt_mask = (tgt_input != train_ds.spy['<pad>'])
      tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))
      tgt_mask = tgt_mask.to(device)
      size = tgt_input.size(1)
      np_mask = torch.triu(torch.ones(size, size)==1).transpose(0,1)
      np_mask = np_mask.float().masked_fill(np_mask == 0, float('-inf')).masked_fill(np_mask == 1, float(0.0))
      np_mask = np_mask.to(device)
      optim.zero_grad()
      preds = model(src.transpose(0,1), tgt_input.transpose(0,1), tgt_mask=np_mask)
      preds = preds.transpose(0,1).contiguous().view(-1, preds.size(-1))
      loss = F.cross_entropy(preds, targets, ignore_index=train_ds.spy['<pad>'], reduction='sum')
      loss.backward()
      optim.step()
      train_loss += loss.item() / 16

      if i % 100 == 0:
        print('Batch {}/{} | Loss: {:.4f}'.format(i, len(train_iter), loss.item() / 16))
        print(greedy_decode_sentence(model, valid_ds[0][0].tolist()))

    print('Epoch {}/{} | Loss: {:.4f}'.format(epoch+1, num_epochs, train_loss/len(train_iter)))

    train_losses.append(train_loss/len(train_iter))

  return train_losses, valid_losses
  """

train_losses, valid_losses = train(train_dl, valid_dl, model, optim, 10)
      
