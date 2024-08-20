import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

from gan.utils.builder import Builders


class NetM(nn.Module):
    def __init__(
        self,
        max_len = 20,
        size_action = 48,

    ):
        super().__init__()
        self.max_len = max_len
        self.size_action = size_action

    def forward(self, x: torch.Tensor):
        # x: (bs, seq_len, 48)

        device = x.device
        bs,seq_len,n_actions = x.shape
        blds = Builders(bs,max_len=seq_len,n_actions=n_actions)

        masks = torch.zeros(bs,seq_len,n_actions).to(device)
        masked_x = torch.zeros(bs,seq_len,n_actions).to(device)
        # prev_select = None
        for i in range(seq_len):

            # get masks
            if i<=self.max_len:
                mask = blds.get_valid_op()# (bs, n_actions)
            else:
                mask = np.zeros([bs, n_actions],dtype=bool)
                mask[:,n_actions-1] = True # the last one is sep

            mask_tensor = torch.from_numpy(mask).to(device)
            masks[:,i,:] = mask_tensor

            # get onehot and push to builders
            logit = x[:,i,:]
            tmp = logit.detach().cpu().numpy()# (bs, n_actions)
            tmp[~mask] = -1e8
            select = tmp.argmax(axis=1)# (bs,)
            # prev_select = select
            assert (mask[:,select]*1.).mean()
            blds.add_token(select)

            # get masked_x
            masked_x[:,i,:] = x[:,i,:]
            masked_x[:,i,:][~mask] = -1e8


        return masked_x,masks,blds
