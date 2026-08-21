import numpy as np
import torch
from torch import Tensor
from typing import Optional
import math


def random_mask(arr_conc, mask_ratio:float=.15):
    '''
    Randomly masking concentration (excluding missing values) for model pre-training.
    arr_conc: a concentration vector of a sample
    mask_ratio: the percentage of concentration tokens to be masked
    '''
    idx_nonzero = np.nonzero(~np.isnan(arr_conc) & (arr_conc != 0))[0]
    n_mask_tokens = int( len(idx_nonzero) * mask_ratio )
    if n_mask_tokens == 0:
        n_mask_tokens = 1
    idx_mask = np.random.choice(idx_nonzero, size=n_mask_tokens, replace=False)
    arr_conc[idx_mask] = np.nan
    return idx_mask, arr_conc


def missing_mask(arr_conc):
    '''
    Masking concentration missing values for model generation.
    arr_conc: a concentration vector of a sample
    '''
    idx_zero_nan = np.nonzero(np.isnan(arr_conc) | (arr_conc == 0))[0]
    idx_mask = idx_zero_nan
    arr_conc[idx_mask] = np.nan
    return idx_mask, arr_conc


def _generate_mask_matrix_VocabFree(
        conc_tokens: Tensor,
        mask_matrix: Tensor,
        num_stable:int=0,
        device: Optional[torch.device] = None
    ) -> Tensor:
    if device is None:
        device = conc_tokens.device

    collect = []
    for i in range(conc_tokens.shape[0]):
        seq_len = mask_matrix.shape[1]

        if num_stable!=0:
            casual_mask = torch.zeros(seq_len+num_stable+1,seq_len+num_stable+1)
            casual_mask[:, torch.where(mask_matrix[i] == 1)[0]+ (num_stable+1)] = 1
            casual_mask[torch.where(mask_matrix[i] == 1)[0]+ (num_stable+1), torch.where(mask_matrix[i] == 1)[0]+ (num_stable+1)] = 0
        else:
            casual_mask = torch.zeros(seq_len,seq_len)
            casual_mask[:, torch.where(mask_matrix[i] == 1)[0]] = 1
            casual_mask[torch.where(mask_matrix[i] == 1)[0], torch.where(mask_matrix[i] == 1)[0]] = 0

        collect.append(casual_mask)

    mask_matrix = torch.stack(collect,dim=0).to(device)
    return mask_matrix


class Mask_Schedule:
    def __init__(self, max_mask_ratio:float=.5, mode:str='train',gamma_func:str='cosine', n_iterations:int=10):
        if mode == 'train':
            self.max_mask_ratio=max_mask_ratio
        elif mode == 'generation':
            self.max_mask_ratio=1.0
        self.mode=mode
        self.f_type=gamma_func
        self.T=n_iterations

    def get_ratio(self, n_samples:Optional[int]=None):
        if self.mode=='train':
            r = torch.rand(n_samples)
            mask_ratio = self._mask_schedule(r)
        elif self.mode=='generation':
            r = torch.linspace(0, 1, self.T+1)
            mask_ratio = self._mask_schedule(r)[:-1]
        else:
            raise TypeError(f"check the mode of Mask_Schedule.get_ratio; {self.mode} is not supported.")
        return mask_ratio

    def _mask_schedule(self, r):
        if self.f_type == "root":
            mask_ratio = 1 - (r ** .5)
        elif self.f_type == "linear":
            mask_ratio = 1 - r
        elif self.f_type == "square":
            mask_ratio = 1 - (r ** 2)
        elif self.f_type == "cosine":
            mask_ratio = torch.cos(r * math.pi * 0.5)
        elif self.f_type == "arccos":
            mask_ratio = torch.arccos(r) / (math.pi * 0.5)
        mask_ratio[mask_ratio<0] = 0
        return mask_ratio * self.max_mask_ratio

