import torch
from torch import nn
from torch import Tensor
from typing import Optional
import math

from .decoder import ImportanceTDL, ImportanceTD


class PositionalEncoding(nn.Module):
  def __init__(self, d_model, dropout=0.1, max_len=5000):
    super(PositionalEncoding, self).__init__()
    self.dropout = nn.Dropout(p=dropout)
    self.d_model = d_model
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0).transpose(0, 1)
    self.register_buffer('pe', pe)

  def forward(self, x):
    x = x * math.sqrt(self.d_model)
    x = x + self.pe[:x.size(0), :]
    return self.dropout(x)


class Transformer(nn.Module):
  def __init__(self, d_model: int = 768, nhead: int = 12, num_encoder_layers: int = 12,
               num_decoder_layers: int = 12, dim_feedforward: int = 3072, dropout: float = 0.1,
               activation: str = "relu", source_vocab_length: int = 32000, target_vocab_length: int = 32000) -> None:
    super(Transformer, self).__init__()
    self.source_embedding = nn.Embedding(source_vocab_length, d_model)
    self.pos_encoder = PositionalEncoding(d_model)
    encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
    encoder_norm = nn.LayerNorm(d_model)
    self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
    self.target_embedding = nn.Embedding(target_vocab_length, d_model)
    decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
    decoder_norm = nn.LayerNorm(d_model)
    self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
    self.out = nn.Linear(d_model, target_vocab_length)
    self._reset_parameters()
    self.d_model = d_model
    self.nhead = nhead

  def forward(self, src: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
          memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
          tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
    if src.size(1) != tgt.size(1):
      raise RuntimeError("the batch number of src and tgt must be equal")
    src = self.source_embedding(src)
    src = self.pos_encoder(src)
    memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
    tgt = self.target_embedding(tgt)
    tgt = self.pos_encoder(tgt)
    output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                          tgt_key_padding_mask=tgt_key_padding_mask,
                          memory_key_padding_mask=memory_key_padding_mask)
    output = self.out(output)
    return output

  def _reset_parameters(self):
    r"""Initiate parameters in the transformer model."""
    for p in self.parameters():
      if p.dim() > 1:
        nn.init.xavier_uniform_(p)


class MyTransformer(nn.Module):
  def __init__(self, d_model: int = 768, nhead: int = 12, num_encoder_layers: int = 12,
               num_decoder_layers: int = 12, dim_feedforward: int = 3072, dropout: float = 0.1,
               activation: str = "relu", source_vocab_length: int = 32000, target_vocab_length: int = 32000,
               add_to_dec: bool = False, yamamoto: bool = False) -> None:
    super(MyTransformer, self).__init__()
    self.source_embedding = nn.Embedding(source_vocab_length, d_model)
    self.pos_encoder = PositionalEncoding(d_model)
    self.importance_embedding = nn.Embedding(2, d_model)
    encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
    encoder_norm = nn.LayerNorm(d_model)
    self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
    self.target_embedding = nn.Embedding(target_vocab_length, d_model)
    decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
    decoder_norm = nn.LayerNorm(d_model)
    self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
    self.out = nn.Linear(d_model, target_vocab_length)
    self._reset_parameters()
    self.d_model = d_model
    self.nhead = nhead
    self.add_to_dec = add_to_dec
    self.yamamoto = yamamoto

  def forward(self, src: Tensor, importance: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
          memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
          tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
    if src.size(1) != tgt.size(1):
      raise RuntimeError("the batch number of src and tgt must be equal")
    src = self.source_embedding(src)
    src = self.pos_encoder(src)
    if not self.add_to_dec:
        src += self.importance_embedding(importance)
    memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
    if self.add_to_dec:
        memory += self.importance_embedding(importance)
    tgt = self.target_embedding(tgt)
    tgt = self.pos_encoder(tgt)
    if self.yamamoto:
        tgt += self.importance_embedding(torch.ones_like(tgt.size()[0], tgt.size()[1]))
    output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                          tgt_key_padding_mask=tgt_key_padding_mask,
                          memory_key_padding_mask=memory_key_padding_mask)
    output = self.out(output)
    return output

  def _reset_parameters(self):
    r"""Initiate parameters in the transformer model."""
    for p in self.parameters():
      if p.dim() > 1:
        nn.init.xavier_uniform_(p)


class MyTransformer2(nn.Module):
  def __init__(self, d_model: int = 768, nhead: int = 12, num_encoder_layers: int = 12,
               num_decoder_layers: int = 12, dim_feedforward: int = 3072, dropout: float = 0.1,
               activation: str = "relu", source_vocab_length: int = 32000, target_vocab_length: int = 32000) -> None:
    super(MyTransformer2, self).__init__()
    self.source_embedding = nn.Embedding(source_vocab_length, d_model)
    self.pos_encoder = PositionalEncoding(d_model)
    encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
    encoder_norm = nn.LayerNorm(d_model)
    self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
    self.target_embedding = nn.Embedding(target_vocab_length, d_model)
    decoder_layer = ImportanceTDL(d_model, nhead, dim_feedforward, dropout, activation)
    decoder_norm = nn.LayerNorm(d_model)
    self.decoder = ImportanceTD(decoder_layer, num_decoder_layers, decoder_norm)
    self.out = nn.Linear(d_model, target_vocab_length)
    self._reset_parameters()
    self.d_model = d_model
    self.nhead = nhead

  def forward(self, src: Tensor, importance: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
          memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
          tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
    if src.size(1) != tgt.size(1):
      raise RuntimeError("the batch number of src and tgt must be equal")
    src = self.source_embedding(src)
    src = self.pos_encoder(src)
    memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
    tgt = self.target_embedding(tgt)
    tgt = self.pos_encoder(tgt)
    output = self.decoder(tgt, memory, importance,
                          tgt_mask=tgt_mask, memory_mask=memory_mask,
                          tgt_key_padding_mask=tgt_key_padding_mask,
                          memory_key_padding_mask=memory_key_padding_mask)
    output = self.out(output)
    return output

  def _reset_parameters(self):
    r"""Initiate parameters in the transformer model."""
    for p in self.parameters():
      if p.dim() > 1:
        nn.init.xavier_uniform_(p)

