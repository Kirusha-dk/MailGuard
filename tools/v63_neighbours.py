"""Stable semantic neighbours, including mixed-label ties at the k boundary."""
import numpy as np


def neighbours(embeddings,anchors,labels,batch_size=256):
    labels=np.asarray(labels)
    if embeddings.shape[1]!=anchors.shape[1] or set(labels)!={0,1}:
        raise ValueError('Matching dimensions and both gold TRAIN classes required')
    # FP32 GEMM accumulation can reorder almost identical neighbours when the
    # caller changes batch size. FP64 plus a fixed score grid removes that noise.
    anchor64=anchors.astype(np.float64);k=min(7,len(anchors));values=[]
    for start in range(0,len(embeddings),batch_size):
        cosine=embeddings[start:start+batch_size].astype(np.float64)@anchor64.T
        cosine=np.round(np.clip(cosine,-1.,1.),8)
        ham=np.max(cosine[:,labels==0],axis=1);spam=np.max(cosine[:,labels==1],axis=1)
        boundary=np.partition(cosine,-k,axis=1)[:,-k]
        above=cosine>boundary[:,None];tied=cosine==boundary[:,None]
        above_count=above.sum(1);tied_count=tied.sum(1)
        # Average all boundary ties rather than choosing an arbitrary subset of
        # otherwise indistinguishable TRAIN messages. Never use TEST neighbours.
        local=(above[:,labels==1].sum(1)+(k-above_count)*tied[:,labels==1].sum(1)/tied_count)/k
        values.append(np.column_stack([ham,spam,spam-ham,local]).astype(np.float32))
    return np.concatenate(values) if values else np.empty((0,4),dtype=np.float32)
