from tqdm import tqdm
from dataloader.dataset import SentencePieceDataset, MyDataset
from model.transformer import Transformer, MyTransformer, MyTransformer2
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
import tensorboardX as tbx
import argparse
import sys
import os
import pickle


def load_ds_and_build_dl(args):
    if args.method == 'conventional':
        train_ds = SentencePieceDataset(
            args.enc_sp_model,
            args.dec_sp_model,
            args.train_data,
            args.max_enc_steps,
            args.max_dec_steps
        )
        valid_ds = SentencePieceDataset(
            args.enc_sp_model,
            args.dec_sp_model,
            args.valid_data,
            args.max_enc_steps,
            args.max_dec_steps
        )
    elif args.method == 'proposed' or args.method == 'attention':
        bin_imp = args.method == 'proposed' and not args.weighted_average
        train_ds = MyDataset(
            args.enc_sp_model,
            args.dec_sp_model,
            args.train_data,
            args.max_enc_steps,
            args.max_dec_steps,
            bin_imp,
            thres=args.imp_thres,
            per_thres=args.per_thres
        )
        valid_ds = MyDataset(
            args.enc_sp_model,
            args.dec_sp_model,
            args.valid_data,
            args.max_enc_steps,
            args.max_dec_steps,
            bin_imp,
            thres=args.imp_thres,
            per_thres=args.per_thres
        )
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_ds.collate_fn
    )
    valid_dl = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=valid_ds.collate_fn
    )
    return train_ds, valid_ds, {'train': train_dl, 'val': valid_dl}


def build_model(args, source_vocab_length, target_vocab_length):
    if args.method == 'conventional':
        model = Transformer(
            d_model=args.d_model,
            nhead=args.nhead,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            dim_feedforward=args.dim_feedforward,
            source_vocab_length=source_vocab_length,
            target_vocab_length=target_vocab_length
        )
    elif args.method == 'proposed':
        model = MyTransformer(
            add_to_dec=args.add_to_dec,
            yamamoto=args.yamamoto,
            weighted=args.weighted_average,
            d_model=args.d_model,
            nhead=args.nhead,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            dim_feedforward=args.dim_feedforward,
            source_vocab_length=source_vocab_length,
            target_vocab_length=target_vocab_length
        )
    elif args.method == 'attention':
        model = MyTransformer2(
            d_model=args.d_model,
            nhead=args.nhead,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            dim_feedforward=args.dim_feedforward,
            source_vocab_length=source_vocab_length,
            target_vocab_length=target_vocab_length
        )
    return model


def greedy_decode_sentence(sp, model, maxlen, ids, importance=None, need_weights=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    indexed = ids
    sentence = torch.autograd.Variable(torch.LongTensor([indexed])).to(device)
    if importance:
        isentence = torch.autograd.Variable(
            torch.LongTensor([importance])
        ).to(device)
    tgt = torch.LongTensor([[sp['<s>']]]).to(device)
    translated_ids = []
    for i in range(maxlen):
        size = tgt.size(0)
        np_mask = torch.triu(torch.ones(size, size)==1).transpose(0,1)
        np_mask = np_mask.float().masked_fill(
            np_mask == 0, float('-inf')
        ).masked_fill(np_mask == 1, float(0.0))
        np_mask = np_mask.to(device)
        if not importance:
            pred = model(sentence.transpose(0, 1), tgt, tgt_mask=np_mask)
        else:
            if need_weights:
                pred, weights = model(
                    sentence.transpose(0, 1),
                    isentence.transpose(0, 1),
                    tgt, tgt_mask=np_mask,
                    need_weights=need_weights
                )
            else:
                pred = model(
                    sentence.transpose(0, 1),
                    isentence.transpose(0, 1),
                    tgt, tgt_mask=np_mask
                )
        add_id = int(pred.argmax(dim=2)[-1])
        translated_ids.append(add_id)
        if add_id == sp['</s>']:
            break
        tgt = torch.cat((tgt, torch.LongTensor([[add_id]]).to(device)))
    if need_weights:
        return translated_ids, weights[0]
    else:
        return translated_ids


def train(args):
    train_ds, valid_ds, dataloaders_dict = load_ds_and_build_dl(args)
    enc_pad_id = train_ds.spx['<pad>']
    dec_pad_id = train_ds.spy['<pad>']
    model = build_model(args, len(train_ds.spx), len(train_ds.spy))
    optim = torch.optim.Adam(
        model.parameters(),
        lr=0.0001, betas=(0.9, 0.98), eps=1e-9
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if args.model_load is not None:
        if not args.tune:
            model.load_state_dict(torch.load(args.model_load))
            optim.load_state_dict(torch.load(args.optim_load))
        else:
            loaded_state_dict = torch.load(args.model_load)
            new_state_dict = model.state_dict().copy()
            for key, value in loaded_state_dict.items():
                new_state_dict[key] = value
            model.load_state_dict(new_state_dict)
            if args.method == 'proposed':
                for param in model.parameters():
                    param.requires_grad = False
                imp_emb_layer = list(model.children())[2]
                for param in imp_emb_layer.parameters():
                    param.requires_grad = True
            elif args.method == 'attention':
                for n_decoder in range(args.num_decoder_layers):
                    for n_linear in range(3):
                        model.state_dict()['decoder.layers.{}.multihead_attn.imp_linears.{}.weight'.format(n_decoder, n_linear)].requires_grad = True
                        model.state_dict()['decoder.layers.{}.multihead_attn.imp_linears.{}.bias'.format(n_decoder, n_linear)].requires_grad = True
                    

    torch.backends.cudnn.benchmark = True

    min_epoch_loss = 1e9
    non_updated_count = 0

    writer = tbx.SummaryWriter(args.log_dir)

    print('started training.')
    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            epoch_loss = 0.0
            optim.zero_grad()

            for i, batch in enumerate(dataloaders_dict[phase]):
                src = batch[0].to(device).long()
                tgt = batch[1].to(device).long()
                if args.method == 'proposed':
                    importance = batch[2].to(device).long()
                elif args.method == 'attention':
                    importance = batch[2].to(device).float()

                with torch.set_grad_enabled(phase == 'train'):
                    tgt_input = tgt[:, :-1]
                    tgt_output = tgt[:, 1:].contiguous().view(-1)

                    src_mask = (src != enc_pad_id)
                    src_mask = src_mask.float().masked_fill(src_mask == 0, float('-inf'))
                    src_mask = src_mask.masked_fill(src_mask == 1, float(0.0))
                    src_mask = src_mask.to(device)

                    tgt_mask = (tgt_input != dec_pad_id)
                    tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf'))
                    tgt_mask = tgt_mask.masked_fill(tgt_mask == 1, float(0.0))
                    tgt_mask = tgt_mask.to(device)

                    size = tgt_input.size(1)
                    np_mask = torch.triu(torch.ones(size, size) == 1).transpose(0, 1)
                    np_mask = np_mask.float().masked_fill(np_mask == 0, float('-inf'))
                    np_mask = np_mask.masked_fill(np_mask == 1, float(0.0))
                    np_mask = np_mask.to(device)

                    if args.method == 'conventional':
                        preds = model(
                            src.transpose(0, 1),
                            tgt_input.transpose(0, 1),
                            tgt_mask=np_mask
                        )
                    elif args.method == 'proposed' or args.method == 'attention':
                        preds = model(
                            src.transpose(0, 1),
                            importance.transpose(0, 1),
                            tgt_input.transpose(0, 1),
                            tgt_mask=np_mask
                        )
                    preds = preds.transpose(0,1).contiguous().view(-1, preds.size(-1))
                    loss = F.cross_entropy(preds, tgt_output, ignore_index=dec_pad_id, reduction='sum')

                    if phase == 'train':
                        loss.backward()
                        if (i + 1) % args.accumulation == 0:
                            optim.step()
                            optim.zero_grad()

                    batch_loss = loss.item() / args.batch_size
                    epoch_loss += batch_loss

                    if phase == 'train':
                        step = epoch * len(dataloaders_dict[phase]) + i
                        writer.add_scalars('loss', {'train_itr': batch_loss}, step)
                        if step % args.interval_decoding == 0:
                            if args.method == 'conventional':
                                writer.add_text(
                                    'val[0]',
                                    valid_ds.spy.DecodeIds(
                                        greedy_decode_sentence(
                                            valid_ds.spy,
                                            model,
                                            args.max_dec_steps,
                                            valid_ds[0][0].tolist()
                                        )
                                    ),
                                    step
                                )
                            elif args.method == 'proposed' or args.method == 'attention':
                                writer.add_text(
                                    'val[0]',
                                    valid_ds.spy.DecodeIds(
                                        greedy_decode_sentence(
                                            valid_ds.spy,
                                            model,
                                            args.max_dec_steps,
                                            valid_ds[0][0].tolist(),
                                            valid_ds[0][2].tolist()
                                        )
                                    ),
                                    step
                                )

            epoch_loss = epoch_loss / len(dataloaders_dict[phase])
            writer.add_scalars('epoch_loss', {phase: epoch_loss}, epoch)

            if phase == 'train':
                optim.step()
                if args.save_all_model:
                    print('saving state dict. [epoch {}]'.format(epoch))
                    dirname = os.path.dirname(args.model_save)
                    if not os.path.exists(dirname):
                        os.mkdir(os.path.dirname(args.model_save))
                    model_path = os.path.join(dirname, "epoch_{}.pt".format(epoch))
                    torch.save(model.state_dict(), model_path)

            elif phase == 'val':
                if epoch_loss < min_epoch_loss:
                    print('updated minimum loss. [epoch {}]'.format(epoch))
                    if not os.path.exists(os.path.dirname(args.model_save)):
                        os.mkdir(os.path.dirname(args.model_save))
                    torch.save(model.state_dict(), args.model_save)
                    if args.optim_save is not None:
                        torch.save(optim.state_dict(), args.optim_save)
                    min_epoch_loss = epoch_loss
                    non_updated_count = 0
                else:
                    non_updated_count += 1
                    if (args.early_stopping
                            and non_updated_count >= args.num_non_updated_counts):
                        print('stopped. [epoch {}]'.format(epoch))
                        writer.flush()
                        return
        writer.flush()
    print('done.')
    return


def split_and_decode(ids, sp):
    sep_id = sp['<sep>']
    decoded = []
    while True:
        if sep_id not in ids:
            decoded.append(sp.DecodeIds(ids))
            return decoded
        else:
            idx = ids.index(sep_id)
            decoded.append(sp.DecodeIds(ids[:idx]))
            if len(ids) - 1 == idx:
                return decoded
            else:
                ids = ids[idx+1:]


def mk_weights_set(spx, spy, article, reference, decoded, weights, importance):
    art_words = [spx.IdToPiece(idx) for idx in article]
    art = [spx.DecodeIds(article)]
    ref = split_and_decode(reference, spy)
    dcd_words = [spy.IdToPiece(idx) for idx in decoded]
    dcd = split_and_decode(decoded, spy)
    weights_set = (art_words, art, ref, dcd_words, dcd, weights, importance)
    return weights_set


def save_summary(path, summaries):
    with open(path, mode="w") as f:
        f.write('\n'.join(summaries))


def test(args):
    if not os.path.exists(args.decode_dir):
        os.mkdir(args.decode_dir)

    if args.method == 'conventional':
        test_ds = SentencePieceDataset(
            args.enc_sp_model,
            args.dec_sp_model,
            args.test_data,
            args.max_enc_steps,
            args.max_dec_steps
        )
        zip_obj = zip(test_ds.x, test_ds.y)
    elif args.method == 'proposed' or args.method == 'attention':
        bin_imp = args.method == 'proposed' and not args.weighted_average
        test_ds = MyDataset(
            args.enc_sp_model,
            args.dec_sp_model,
            args.test_data,
            args.max_enc_steps,
            args.max_dec_steps,
            bin_imp,
            args.imp_thres,
            per_thres=args.per_thres
        )
        zip_obj = zip(test_ds.x, test_ds.y, test_ds.z)

    model = build_model(args, len(test_ds.spx), len(test_ds.spy))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load(args.model_load))

    torch.backends.cudnn.benchmark = True

    for i, data in enumerate(tqdm(zip_obj, total=len(test_ds.x))):
        if args.method == 'conventional':
            decoded = greedy_decode_sentence(
                test_ds.spy,
                model,
                args.max_dec_steps,
                data[0].tolist()
            )
        elif args.method == 'proposed' or args.method == 'attention':
            if args.need_weights:
                decoded, weights = greedy_decode_sentence(
                    test_ds.spy,
                    model,
                    args.max_dec_steps,
                    data[0].tolist(), data[2].tolist(),
                    need_weights=args.need_weights
                )
            else:
                decoded = greedy_decode_sentence(
                    test_ds.spy,
                    model,
                    args.max_dec_steps,
                    data[0].tolist(), data[2].tolist()
                )
        if args.need_weights:
            weights_set = mk_weights_set(test_ds.spx, test_ds.spy, data[0].tolist(), data[1].tolist(), decoded, weights.tolist(), data[2].tolist())
            with open(os.path.join(args.decode_dir+'-weight', str(i).zfill(len(str(len(test_ds.x)))))+".pickle", mode="wb") as f:
                pickle.dump(weights_set, f)
        
        decoded_summaries = split_and_decode(decoded, test_ds.spy)
        save_summary(
            os.path.join(
                args.decode_dir, str(i).zfill(len(str(len(test_ds.x))))
            ) + ".txt",
            decoded_summaries
        ) 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], required=True)
    parser.add_argument(
        '--method',
        choices=['conventional', 'proposed', 'attention'],
        default='conventional'
    )
    # train, val, testデータとかはmodeによって必要かどうかが変わる
    # conds[0]: for train and val, conds[1]: for test
    conds = [False, False]
    if '--mode' in sys.argv:
        index = sys.argv.index('--mode')
        if len(sys.argv) > index + 1:
            if sys.argv[index + 1] == 'train':
                conds[0] = True
            elif sys.argv[index + 1] == 'test':
                conds[1] = True
    parser.add_argument(
        '--tune',
        action='store_true',
        help='load and tune the trained model'
    )
    parser.add_argument(
        '--add-to-dec',
        action='store_true',
        help='add importance embedding to decoder\'s memory'
    )
    parser.add_argument(
        '--yamamoto',
        action='store_true',
        help='yamamoto option'
    )
    parser.add_argument('--epochs', type=int, required=conds[0])
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--batch-size', type=int, required=conds[0])
    parser.add_argument(
        '--accumulation',
        type=int,
        default=1,
        help='backfoward every [accumulation] times'
    )
    parser.add_argument(
        '--interval-decoding',
        type=int,
        default=1024,
        help='decode interval for development data[0] while training'
    )
    parser.add_argument('--early-stopping', action='store_true')
    parser.add_argument(
        '--num-non-updated-counts',
        type=int,
        required='--early-stopping' in sys.argv,
        help='number of non-updated times before stopping'
    )
    parser.add_argument('--save-all-model', action='store_true')
    parser.add_argument('--d-model', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num-encoder-layers', type=int, default=6)
    parser.add_argument('--num-decoder-layers', type=int, default=6)
    parser.add_argument('--dim-feedforward', type=int, default=2048)
    parser.add_argument('--max-enc-steps', type=int, default=400)
    parser.add_argument('--max-dec-steps', type=int, default=100)
    parser.add_argument('--enc-sp-model', required=True)
    parser.add_argument('--dec-sp-model', required=True)
    parser.add_argument('--train-data', required=conds[0])
    parser.add_argument('--valid-data', required=conds[0])
    parser.add_argument('--test-data', required=conds[1])
    parser.add_argument(
        '--model-load',
        required=conds[1] or conds[0] and '--optim-load' in sys.argv,
        help='the model path that is loaded'
    )
    parser.add_argument(
        '--model-save',
        required=conds[0],
        help='the model path that is saved'
    )
    parser.add_argument(
        '--optim-load',
        required=conds[0] and '--model-load' in sys.argv and '--tune' not in sys.argv,
        help='the optimizer path that is loaded berfore training'
    )
    parser.add_argument(
        '--optim-save',
        help='the optimizer path that is saved'
    )
    parser.add_argument('--log-dir', required=conds[0])
    parser.add_argument('--decode-dir', required=conds[1])
    parser.add_argument('--need-weights', action='store_true')
    parser.add_argument('--weighted-average', action='store_true')
    parser.add_argument('--imp-thres', type=int, default=3, help='importance threshold')
    parser.add_argument('--per-thres', type=float)

    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)


if __name__ == "__main__":
    main()
