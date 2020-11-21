import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np

class MILoss(nn.Module):
    def __int__(self):
        super(MILoss, self).__int__()

    def cosine_sim(self, x1, x2, dim=1, eps=1e-8):
        ip = torch.mm(x1, x2.t())
        w1 = torch.norm(x1, 2, dim)
        w2 = torch.norm(x2, 2, dim)
        return ip / torch.ger(w1,w2).clamp(min=eps)

    def get_lower_bound(self, probs, target,anchor_bs):
        same_idx = target.view(-1).nonzero(as_tuple=False)
        select_same = probs.view(-1)[same_idx]
        joint = select_same + np.log(anchor_bs)
        marginal = torch.logsumexp(probs, dim=1)
        lower_bound = torch.mean(joint - marginal)
        # lower_bound = 1 / (1 + torch.exp(lower_bound))
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
        margin = -0.2
        anchor_bs = img_vec.size(0)
        target=torch.eye(anchor_bs).cuda()
        probs = torch.abs(self.cosine_sim(img_vec, vox_vec).reshape(anchor_bs, -1)) * S - M 
        lower_bound = self.get_lower_bound(probs, target, anchor_bs)
        # upper_bound = self.get_upper_bound(probs, target, anchor_bs)
        # mutual_loss = upper_bound - lower_bound
        # mutual_loss = torch.nn.functional.relu(mutual_loss + margin)
        # logging.info('Upper = %.3f , Lower = %.3f , MILoss = %.3f' %
        #              (upper_bound, lower_bound, mutual_loss))

        return -lower_bound
