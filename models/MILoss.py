import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
import logging
import pdb

class MILoss(nn.Module):
    def __int__(self):
        super(MILoss, self).__int__()

    def cosine_sim(self, x1, x2, dim=1, eps=1e-8):
        ip = torch.mm(x1, x2.t())
        w1 = torch.norm(x1, 2, dim)
        w2 = torch.norm(x2, 2, dim)
        return ip / torch.ger(w1,w2).clamp(min=eps)

    def get_lower_bound(self, probs, target, anchor_bs):
        same_idx = target.view(-1).nonzero(as_tuple=False)
        select_same = probs.view(-1)[same_idx]
        joint = select_same + np.log(anchor_bs)
        marginal = torch.logsumexp(probs, dim=1)
        lower_bound = torch.mean(joint - marginal)
        return lower_bound

    def get_upper_bound(self, probs, target, anchor_bs):
        same_idx = target.view(-1).nonzero(as_tuple=False)
        diff_idx = (1-target).view(-1).nonzero(as_tuple=False)
        select_same = probs.view(-1)[same_idx]
        joint = select_same + np.log(anchor_bs-1)
        select_diff = probs.view(-1)[diff_idx].reshape(probs.size(0),-1)
        marginal = torch.logsumexp(select_diff, dim=1)
        upper_bound = torch.mean(joint - marginal)
        return upper_bound

    def forward(self, img_vec, vox_vec):
        S = 2.0
        M = 2.0
        # margin = -0.2
        margin = 0
        anchor_bs = img_vec.size(0) # K

        probs = torch.abs(self.cosine_sim(img_vec, vox_vec).reshape(anchor_bs, -1)) * S - M

        target_1 = torch.eye(anchor_bs).cuda()
        probs_11 = probs[:anchor_bs//2,:anchor_bs//2].contiguous()
        probs_12 = probs[anchor_bs//2:anchor_bs,anchor_bs//2:anchor_bs].contiguous()

        target_2 = torch.zeros(anchor_bs//2, anchor_bs//2 + 1).cuda()
        target_2[:,anchor_bs//2] = 1
        probs_22 = probs[anchor_bs//2:anchor_bs,:anchor_bs//2].contiguous()
        probs_21 = probs[:anchor_bs//2,anchor_bs//2:anchor_bs].contiguous()
        idx = torch.eye(anchor_bs//2)
        idx_c = idx.clone()
        idx[:-1]= idx_c[1:]
        idx[-1] = idx_c[0]
        rand_idx = idx.view(-1).nonzero(as_tuple=False)
        rand_1 = probs_11.view(-1)[rand_idx]
        rand_2 = probs_12.view(-1)[rand_idx]
        probs_21 = torch.cat((probs_21, rand_1), 1)
        probs_22 = torch.cat((probs_22, rand_2), 1)

        # probs.shape = [K, K]
        lower_bound_11 = self.get_lower_bound(probs, target_1, anchor_bs)
        # lower_bound_12 = self.get_lower_bound(probs, target_1, anchor_bs)
        # lower_bound_1 = (lower_bound_11 + lower_bound_12)/2
        lower_bound_1 = lower_bound_11
        lower_bound_21 = self.get_lower_bound(probs_21, target_2, anchor_bs//2+1)
        lower_bound_22 = self.get_lower_bound(probs_22, target_2, anchor_bs//2+1)
        lower_bound_2 = (lower_bound_21 + lower_bound_22)/2
        upper_bound_21 = self.get_upper_bound(probs_21, target_2, anchor_bs//2+1)
        upper_bound_22 = self.get_upper_bound(probs_22, target_2, anchor_bs//2+1)
        upper_bound_2 = (upper_bound_21 + upper_bound_22)/2

        constraint = torch.nn.functional.relu(upper_bound_2 - lower_bound_1 + margin)
        mutual_loss = lower_bound_1 + lower_bound_2 - constraint
        # mutual_loss = lower_bound_1 + lower_bound_2
        logging.info('Lower_1 = %.3f , Lower_2 = %.3f, Constraint = %.3f, MILoss = %.3f' %
                     (lower_bound_1, lower_bound_2, constraint, mutual_loss))

        return -mutual_loss