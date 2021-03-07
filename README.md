# MyTransformer
## About
文書要約のタスクにおいて，Transformerを改善することで良い要約を生成できるようにしようとしているやつ．入力トークンのembeddingに，その単語が属する文が文書中において重要であるかを意味するembeddingを加算することで精度向上を図る．

![imp_emb](https://user-images.githubusercontent.com/39112867/110201196-3216f100-7ea5-11eb-980f-78af3ef87e5d.png)

## Required
- [PyTorch](https://pytorch.org/)
- [tensorboardX](https://github.com/lanpa/tensorboardX)
- [sentencepiece](https://github.com/google/sentencepiece)
- [tqdm](https://github.com/tqdm/tqdm)

## Usage
```
$ python main.py ...
```

### Options
| オプション名               | 説明                                                                           | デフォルト | 必須？                                                   |
| -------------------------- | ------------------------------------------------------------------------------ | ---------- | -------------------------------------------------------- |
| `--mode`                   | 学習かテストか．`train` or `test`                                              |            | ○                                                       |
| `--method`                 | 普通のtransformerか提案モデルか．`conventional` or `proposed`                  |            | ○                                                       |
| `--tune`   | 指定した場合，`--method`: `proposed`時に，学習済みの`conventional`モデルを読み込んで重要性を表すembeddingのみfine tuningする  |   |   |
| `--epochs`                 | 学習エポック数                                                                 |            | ○                                                       |
| `--start-epoch`            | 開始エポック                                                                   | 0          |                                                          |
| `--batch-size`             | バッチサイズ                                                                   |            | ○                                                       |
| `--accumulation`           | Gradient Accumulationにおけるパラメータ更新までのバッチの回数(?)               | 1          |                                                          |
| `--interval-decoding`      | 何バッチおきにvalid[0]をデコードしてlog出力するか                              | 1024       |                                                          |
| `--early-stopping`         | 指定するとearly stoppingされる                                                 |            |                                                          |
| `--num-non-updated-counts` | 検証データのエポックロスが何回最低値を更新しなかった場合にearly stoppingするか |            | `--early-stopping`指定時は必須                           |
| `--d-model`                | embedding size                                                                 | 512        |                                                          |
| `--nhead`                  | head数                                                                         | 8          |                                                          |
| `--num-encoder-layers`     | encoderレイヤ数                                                                | 6          |                                                          |
| `--num-decoder-layers`     | decoderレイヤ数                                                                | 6          |                                                          |
| `--dim-feedforward`        | feedforward次元数                                                              | 2048       |                                                          |
| `--max-enc-steps`          | 最大入力系列長                                                                 | 400        |                                                          |
| `--max-dec-steps`          | 最大出力系列長                                                                 | 100        |                                                          |
| `--enc-sp-model`           | 入力系列のsentencepieceモデルパス                                              |            | ○                                                       |
| `--dec-sp-model`           | 出力系列のsentencepieceモデルパス                                              |            | ○                                                       |
| `--train-data`             | 学習データのpickleファイルパス                                                 |            | `--mode`で`train`指定時に必須                            |
| `--valid-data`             | 検証データのpickleファイルパス                                                 |            | `--mode`で`train`指定時に必須                            |
| `--test-data`              | テストデータのpickleファイルパス                                               |            | `--mode`で`test`指定時に必須                             |
| `--model-load`             | 学習済みモデルパラメータを読み込む際のモデルパス                               |            | `--mode`で`test`指定時，`--optim-load`指定時，または`--tune`指定時に必須 |
| `--model-save`             | モデルパラメータの保存先                                                       |            | `--mode`で`train`指定時に必須                            |
| `--optim-load`             | オプティマイザを読み込む際のパス                                               |            | `--model-load`指定時に必須．ただし`--tune`指定時は必須ではない                               |
| `--optim-save`             | オプティマイザの保存先                                                         |            |                                                          |
| `--log-dir`                | tensorboardのlog出力先                                                         |            | `--mode`で`train`指定時に必須                            |
| `--decode-dir`             | テストデータのデコード先ディレクトリ                                           |            | `--mode`で`test`指定時に必須                             |

学習時，テスト時それぞれで`train.sh.sample`，`test.sh.sample`を用いるとよい．

## 入力データのpickle構造
`N:=len(dataset)`とした時，`N×3`の多次元配列
```
pickle-data
    └─ list[N]
         ├─ data 0
         │    ├ src
         │    ├ target
         │    └ importance
         ├─ data 1
         │    ├ src
         │    ├ target
         │    └ importance
         │
         ├─ ...
         │
         └─ data N-1
              ├ src
              ├ target
              └ importance
```

### src
入力系列をsentencepieceでEncodeしたID系列．文頭を`<cls>`，文間を`<sep>`とする．文末は特に何もつけなくてよい．

### target
出力系列をsentencepieceでEncodeしたID系列．文頭を`<s>`，文間を`<sep>`，文末を`</s>`とする．testデータとしてpickleを読み込む際にはtargetは不要．

### importance
srcの対応するポジションの文が，srcの全て文の中で何番目に重要であるかを整数型で格納する．`<cls>`など特殊トークンに対応する箇所には-1を格納する．例えばsrcが
```
['<cls>', 101, 102, '<sep>', 103, 104, '<sep>', 105, 106]
```
となっており，2文目，3文目，1文目の順に重要であるとする場合，importanceは
```
[-1, 2, 2, -1, 0, 0, -1, 1, 1]
```
となるようにする．`--method`が`conventional`の場合importanceは不要．
