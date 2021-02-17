from collections import OrderedDict
from tqdm import tqdm
from dataloader.dataset import SentencePieceDataset, MyDataset
from model.transformer import Transformer, MyTransformer
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F


BATCH_SIZE = 16
MAGNIFICATION = 4


train_ds = SentencePieceDataset('../data/sp/article.model',
                                '../data/sp/summary.model',
                                '../data/ds/train.pickle',
                                400, 100)
valid_ds = SentencePieceDataset('../data/sp/article.model',
                                '../data/sp/summary.model',
                                '../data/ds/valid.pickle',
                                400, 100)
test_ds = SentencePieceDataset('../data/sp/article.model',
                               '../data/sp/summary.model',
                               '../data/ds/test.pickle',
                               400, 10000)
# train_ds = MyDataset('../data/sp/article.model',
#                      '../data/sp/summary.model',
#                      '../data/ds/train.pickle',
#                      400, 100)
# valid_ds = MyDataset('../data/sp/article.model',
#                      '../data/sp/summary.model',
#                      '../data/ds/valid.pickle',
#                      400, 100)
# test_ds = MyDataset('../data/sp/article.model',
#                     '../data/sp/summary.model',
#                     '../data/ds/test.pickle',
#                     400, 10000)

train_dl = DataLoader(train_ds,
                      batch_size=BATCH_SIZE,
                      shuffle=True,
                      collate_fn=train_ds.collate_fn)
valid_dl = DataLoader(valid_ds,
                      batch_size=BATCH_SIZE,
                      shuffle=False,
                      collate_fn=valid_ds.collate_fn)


model = Transformer(d_model = 512,#768,
                    nhead = 8,#12,
                    num_encoder_layers = 8,#12,
                    num_decoder_layers = 8,#12,
                    dim_feedforward = 2048,#3072,
                    source_vocab_length=len(test_ds.spx),
                    target_vocab_length=len(test_ds.spy))
# model = MyTransformer(d_model = 512, # 768
#                       nhead = 8, # 12
#                       num_encoder_layers = 6, # 12
#                       num_decoder_layers = 6, # 12
#                       dim_feedforward = 2048, # 3072
#                       source_vocab_length=len(test_ds.spx),
#                       target_vocab_length=len(test_ds.spy))

optim = torch.optim.Adam(model.parameters(), lr=0.0001,
                         betas=(0.9, 0.98), eps=1e-9)


def greedy_decode_sentence(model, ids, importance=None):
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model.eval()
  indexed = ids
  sentence = torch.autograd.Variable(torch.LongTensor([indexed])).to(device)
  if importance:
    isentence = torch.autograd.Variable(torch.LongTensor([importance])).to(device)
  tgt = torch.LongTensor([[test_ds.spy['<s>']]]).to(device)
  translated_ids = []
  maxlen = 100
  for i in range(maxlen):
    size = tgt.size(0)
    np_mask = torch.triu(torch.ones(size, size)==1).transpose(0,1)
    np_mask = np_mask.float().masked_fill(np_mask == 0, float('-inf')).masked_fill(np_mask == 1, float(0.0))
    np_mask = np_mask.to(device)
    if not importance:
      pred = model(sentence.transpose(0, 1), tgt, tgt_mask=np_mask)
    else:
      pred = model(sentence.transpose(0, 1),
                   isentence.transpose(0, 1),
                   tgt, tgt_mask=np_mask)
    add_id = int(pred.argmax(dim=2)[-1])
    translated_ids.append(add_id)
    if add_id == test_ds.spy['</s>']:
      break
    tgt = torch.cat((tgt, torch.LongTensor([[add_id]]).to(device)))
  # return test_ds.spy.DecodeIds(translated_ids)
  return translated_ids


def train(train_iter, val_iter, model, optim, num_epochs, is_mine=False):
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model.to(device)
  # model.load_state_dict(torch.load('checkpoints/small-batch-16/checkpoint_best_epoch.pt'))
  # optim.load_state_dict(torch.load('checkpoints/small-batch-16/optimizer.pt'))
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

      # with tqdm(dataloaders_dict[phase]) as pbar:
        # pbar.set_description("Epoch {}/{} | {:^5}".format(epoch+1, num_epochs, phase))
      optim.zero_grad()
      for i, batch in enumerate(dataloaders_dict[phase]):
        src = batch[0].to(device).long()
        tgt = batch[1].to(device).long()
        if is_mine:
          importance = batch[2].to(device).long()

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

          if not is_mine:
            preds = model(src.transpose(0, 1),
                          tgt_input.transpose(0, 1),
                          tgt_mask=np_mask)
          else:
            preds = model(src.transpose(0, 1),
                          importance.transpose(0, 1),
                          tgt_input.transpose(0, 1),
                          tgt_mask=np_mask)
          preds = preds.transpose(0,1).contiguous().view(-1, preds.size(-1))
          loss = F.cross_entropy(preds, tgt_output, ignore_index=train_ds.spy['<pad>'], reduction='sum')

          if phase == 'train':
            loss.backward()
            if (i + 1) % MAGNIFICATION == 0:
              optim.step()
              optim.zero_grad()

          batch_loss = loss.item() / BATCH_SIZE
          epoch_loss += batch_loss

          if phase == 'train' and i % (MAGNIFICATION * 8) == 0:
            print('Epoch {}/{} | Batch {}/{} | {:^5} | Loss: {:.4f}'.format(epoch+1,
                                                                              num_epochs,
                                                                              i+1,
                                                                              len(dataloaders_dict[phase]),
                                                                              phase,
                                                                              batch_loss))
            if i % (MAGNIFICATION * 32) == 0:
              if not is_mine:
                print(valid_ds.spy.DecodeIds(greedy_decode_sentence(model, valid_ds[0][0].tolist())))
              else:
                print(valid_ds.spy.DecodeIds(greedy_decode_sentence(model, valid_ds[0][0].tolist(), valid_ds[0][2].tolist())))


           # pbar.set_postfix(OrderedDict(loss='{:.4f}'.format(batch_loss)))

      epoch_loss = epoch_loss / len(dataloaders_dict[phase])
      print('Epoch {}/{} | {:^5} | Loss: {:.4f}'.format(epoch+1,
                                                        num_epochs,
                                                        phase,
                                                        epoch_loss))

      if phase == 'val':
        if epoch_loss < min(losses_dict['val'], default=1e9):
          print("saving state dict")
          torch.save(model.state_dict(), f"checkpoint_best_epoch.pt")
          torch.save(optim.state_dict(), "optimizer.pt")

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


def test(model, is_mine=False):
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model.to(device)
  model.load_state_dict(torch.load('checkpoint_best_epoch.pt'))

  torch.backends.cudnn.benchmark = True

  reference_dir = "test/reference/"
  decoded_dir = "test/decoded/"

  if is_mine:
    zip_obj = zip(test_ds.x, test_ds.y, test_ds.z)
  else:
    zip_obj = zip(test_ds.x, test_ds.y)

  for i, data in enumerate(tqdm(zip_obj)):
    reference_summaries = split_and_decode(data[1].tolist())
    save_summary(reference_dir + str(i).zfill(5) + ".txt", reference_summaries)
    if is_mine:
      decoded = greedy_decode_sentence(model, data[0].tolist(), data[2].tolist())
    else:
      decoded = greedy_decode_sentence(model, data[0].tolist())
    decoded_summaries = split_and_decode(decoded)
    save_summary(decoded_dir + str(i).zfill(5) + ".txt", decoded_summaries)


def restore_importance(ds):
  importance_dir = "test/importance/"
  for i, (x, y, z) in enumerate(tqdm(ds)):
    ids = []
    pre = 0
    for xx, zz in zip(x, z):
      if zz == 1:
        ids.append(xx.tolist())
      elif pre == 1:
        ids.append(ds.spy['<sep>'])
      pre = zz
    summaries = split_and_decode(ids)
    save_summary(importance_dir + str(i).zfill(5) + ".txt", summaries)
    

train_losses, valid_losses = train(train_dl, valid_dl, model, optim, 12, False)
print(train_losses)
print(valid_losses)
# test(model, is_mine=True)
# restore_importance(test_ds)


