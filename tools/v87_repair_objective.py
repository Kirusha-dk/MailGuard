"""Operating-point-aware adaptation and monotonic presentation accounting."""
import math
import torch


def threshold_aware_loss(logits, labels, old_replay, cutoff=.04897100105881691):
    if not 0 < cutoff < 1 or logits.shape != labels.shape or logits.shape != old_replay.shape:
        raise ValueError('Invalid recovery objective inputs')
    offset=math.log(cutoff/(1-cutoff))
    effective=torch.where(old_replay,logits,logits-offset)
    cost=torch.where((~old_replay)&(labels==1),1.5,1.)
    return (torch.nn.functional.binary_cross_entropy_with_logits(effective,labels,reduction='none')*cost).mean()


def roll_forward_exposures(counts, deltas):
    """Restoring earlier weights never restores an earlier usage allowance."""
    result=dict(counts)
    for d in deltas:
        if (result.get(d['identity'],0)!=d['before'] or
                d['after']!=d['before']+1 or not 0<=d['before']<d['after']<=3):
            raise ValueError('Replayed, reset or exhausted presentation ledger')
        result[d['identity']]=d['after']
    return result
