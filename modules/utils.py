import torch
from typing import Optional
from torch import nn

def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src

def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)

def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    return scatter_sum(src, index, dim, out, dim_size)


class EdgelistDrop(nn.Module):
    def __init__(self):
        super(EdgelistDrop, self).__init__()

    def forward(self, edgeList, keep_rate=None, return_mask=False, **kwargs):
        # 兼容 GraphProCCF 的参数名 keep_prob
        if keep_rate is None:
            keep_rate = kwargs.get('keep_prob', 1.0)

        # 完全保留原行为
        if keep_rate == 1.0:
            mask = torch.ones(edgeList.size(0), dtype=torch.bool, device=edgeList.device)
            return (edgeList, mask) if return_mask else edgeList

        edgeNum = edgeList.size(0)

        # 按你原来的写法:
        mask = (torch.rand(edgeNum, device=edgeList.device) + keep_rate).floor().bool()
        newEdgeList = edgeList[mask]

        return (newEdgeList, mask) if return_mask else newEdgeList

class SpAdjEdgeDrop(nn.Module):
    def __init__(self):
        super(SpAdjEdgeDrop, self).__init__()

    def forward(self, adj, keep_rate, return_mask=False):
        if keep_rate == 1.0:
            return adj
        vals = adj._values()
        idxs = adj._indices()
        edgeNum = vals.size()
        mask = (torch.rand(edgeNum) + keep_rate).floor().type(torch.bool)
        newVals = vals[mask]  # / keep_rate
        newIdxs = idxs[:, mask]
        if return_mask:
            return torch.sparse.FloatTensor(newIdxs, newVals, adj.shape), mask
        else:
            return torch.sparse.FloatTensor(newIdxs, newVals, adj.shape)


def reg_params(model):
    reg_loss = 0
    for W in model.parameters():
        reg_loss += W.norm(2).square()
    return reg_loss
