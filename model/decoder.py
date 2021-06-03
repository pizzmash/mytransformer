import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional, Tuple, List
import math


Tensor = torch.Tensor


def _in_projection_packed(
	q: Tensor,
	k: Tensor,
	v: Tensor,
	w: Tensor,
	b: Optional[Tensor] = None,
) -> List[Tensor]:
	E = q.size(-1)
	w_q, w_kv = w.split([E, E * 2])
	if b is None:
		b_q = b_kv = None
	else:
		b_q, b_kv = b.split([E, E * 2])
	return (F.linear(q, w_q, b_q),) + F.linear(k, w_kv, b_kv).chunk(2, dim=-1)


def _scaled_dot_product_attention(
	q: Tensor,
	k: Tensor,
	v: Tensor,
	i: Tensor,
	attn_mask: Optional[Tensor] = None,
	dropout_p: float = 0.0
) -> Tuple[Tensor, Tensor]:
	B, Nt, E = q.shape
	q = q/ math.sqrt(E)
	attn = torch.bmm(q, k.transpose(-2, -1))
	# print(attn.size())
	attn = attn * i.view(B, 1, i.size(-1))
	if attn_mask is not None:
		attn += attn_mask
	attn = F.softmax(attn, dim=-1)
	if dropout_p > 0.0:
		attn = F.dropout(attn, p=dropout_p)
	output = torch.bmm(attn, v)
	return output, attn


def importance_mha_forward(
	query: Tensor,
	key: Tensor,
	value: Tensor,
	importance_weights: Tensor,
	embed_dim_to_check: int,
	num_heads: int,
	in_proj_weight: Tensor,
	in_proj_bias: Optional[Tensor],
	dropout_p: float,
	out_proj_weight: Tensor,
	out_proj_bias: Optional[Tensor],
	training: bool = True,
	key_padding_mask: Optional[Tensor] = None,
	need_weights: bool = True,
	attn_mask: Optional[Tensor] = None
) -> Tuple[Tensor, Optional[Tensor]]:
	# set up shape vars
	tgt_len, bsz, embed_dim = query.shape
	src_len, _, _ = key.shape
	assert embed_dim == embed_dim_to_check, \
		f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
	if isinstance(embed_dim, torch.Tensor):
		# embed_dim can be a tensor when JIT tracing
		head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
	else:
		head_dim = embed_dim // num_heads
	assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
	assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

	#
	# compute in-projection
	#
	q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
	
	# prep attention mask
	if attn_mask is not None:
		if attn_mask.dtype == torch.uint8:
			warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
			attn_mask = attn_mask.to(torch.bool)
		else:
			assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, \
				f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
		# ensure attn_mask's dim is 3
		if attn_mask.dim() == 2:
			correct_2d_size = (tgt_len, src_len)
			if attn_mask.shape != correct_2d_size:
				raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
			attn_mask = attn_mask.unsqueeze(0)
		elif attn_mask.dim() == 3:
			correct_3d_size = (bsz * num_heads, tgt_len, src_len)
			if attn_mask.shape != correct_3d_size:
				raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
		else:
			raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

	# prep key padding mask
	if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
		warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
		key_padding_mask = key_padding_mask.to(torch.bool)


	#
	# reshape q, k, v for multihead attention and make em batch first
	#
	q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
	k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
	v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
	importance_weights = importance_weights.contiguous().transpose(0, 1)
	buf = torch.empty(bsz * num_heads, importance_weights.size(-1))
	for pos in range(len(buf)):
		buf[pos] = importance_weights[int(pos / num_heads)]
	importance_weights = buf.to(importance_weights.dtype).to(importance_weights.device)

	# update source sequence length after adjustments
	src_len = k.size(1)

	# merge key padding and attention masks
	if key_padding_mask is not None:
		assert key_padding_mask.shape == (bsz, src_len), \
			f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
		key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len). \
			expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
		if attn_mask is None:
			attn_mask = key_padding_mask
		elif attn_mask.dtype == torch.bool:
			attn_mask = attn_mask.logical_or(key_padding_mask)
		else:
			attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

	# convert mask to float
	if attn_mask is not None and attn_mask.dtype == torch.bool:
		new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
		new_attn_mask.masked_fill_(attn_mask, float("-inf"))
		attn_mask = new_attn_mask

	# adjust dropout probability
	if not training:
		dropout_p = 0.0

	#
	# (deep breath) calculate attention and out projection
	#
	attn_output, attn_output_weights = _scaled_dot_product_attention(q, k, v, importance_weights, attn_mask, dropout_p)
	attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
	attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

	if need_weights:
		# average attention weights over heads
		attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
		return attn_output, attn_output_weights.sum(dim=1) / num_heads
	else:
		return attn_output, None

class ImportanceMHA(nn.MultiheadAttention):
	def __init__(self, embed_dim, num_heads, dropout=0.1, num_imp_linear=3):
		super(ImportanceMHA, self).__init__(embed_dim, num_heads, dropout=dropout, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None)
		self.imp_linears = nn.ModuleList([nn.Linear(1, 1).cuda() for _ in range(num_imp_linear)])
		self._reset_parameters()

	def forward(self, query: Tensor, key: Tensor, value: Tensor, imp: Tensor, key_padding_mask: Optional[Tensor] = None, need_weights: bool = True, attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
		iw = imp.contiguous().view(-1, 1).to(torch.float32)
		bs = imp.size(1)
		###
		iw = F.relu(iw)
		iw = iw / torch.max(iw) * -1. + 1.
		###
		# for i, f in enumerate(self.imp_linears):
		#	if i < len(self.imp_linears) - 1:
		#		iw = F.relu(f(iw))
		#	else:
		#		iw = f(iw)
		###
		iw = iw.contiguous().view(-1, bs)
		return importance_mha_forward(
			query, key, value, iw,
			self.embed_dim, self.num_heads,
			self.in_proj_weight, self.in_proj_bias,
			self.dropout,
			self.out_proj.weight, self.out_proj.bias,
			training=self.training,
			key_padding_mask=key_padding_mask, need_weights=need_weights,
			attn_mask=attn_mask)


class ImportanceTDL(nn.TransformerDecoderLayer):
	def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
		super(ImportanceTDL, self).__init__(d_model, nhead, dim_feedforward, dropout, activation)
		self.multihead_attn = ImportanceMHA(d_model, nhead, dropout=dropout)

	def forward(self, tgt: Tensor, memory: Tensor, importance: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
		tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
		tgt = tgt + self.dropout1(tgt2)
		tgt = self.norm1(tgt)
		tgt2 = self.multihead_attn(tgt, memory, memory, importance, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
		tgt = tgt + self.dropout2(tgt2)
		tgt = self.norm2(tgt)
		tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
		tgt = tgt + self.dropout3(tgt2)
		tgt = self.norm3(tgt)
		return tgt


class ImportanceTD(nn.TransformerDecoder):
	def forward(self, tgt: Tensor, memory: Tensor, importance: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
		output = tgt
		for mod in self.layers:
			output = mod(output, memory, importance, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
		if self.norm is not None:
			output = self.norm(output)
		return output


if __name__ == "__main__":
	device = torch.device("cuda:0")
	model = ImportanceMHA(10, 2, dropout=0.1).to(device)
	q = torch.randn(5, 2, 10).to(device)
	kv = torch.randn(3, 2, 10).to(device)
	i = torch.randn(3, 2).to(device)
	model(q, kv, kv, i, need_weights=False)
