"""Frozen multilingual ONNX encoder and TRAIN-only semantic neighbours."""
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
import numpy as np
from train_v55_multilingual import file_sha

MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
REVISION = 'e8f8c211226b894fcb81acc59f3b34ba3efd5f42'
ONNX = 'onnx/model_quint8_avx2.onnx'
VIEW_VERSION = 'v61-pair-head-tail-128-v1'


def group(source):
    return 'chinese' if source == 'TREC06-Chinese' else 'recent' if source.startswith('SpamArchive-2026-') else 'archive' if source.startswith('SpamArchive-') else 'legacy'


def training_pool(rows):
    buckets = defaultdict(list); weak = []
    for row in rows:
        if row.get('weakLabel'): weak.append(row)
        else: buckets[group(row['source']),row['y']].append(row)
    chosen = list(weak)
    for (name,y),values in sorted(buckets.items()):
        cap = 12000 if y == 0 and name == 'legacy' else 6000
        chosen += sorted(values,key=lambda r:r['identity'])[:cap]
    return sorted(chosen,key=lambda r:r['identity'])


def windows(row):
    # The contextual encoder sees authored words rather than source/year/sender
    # identifiers. The tokenizer shares its token budget between head and tail.
    focus = row['focus']; context = row['context']
    head = re.sub(r'\bhosts\s+.*?\s+start\s+', ' ',focus,count=1)
    return head[:1000], context[-1000:]


def script(row):
    letters = [c for c in row['context'] if c.isalpha()]
    cjk = sum('\u3040'<=c<='\u30ff' or '\u3400'<=c<='\u9fff' or '\uac00'<=c<='\ud7af' for c in letters)
    nonlatin = sum(ord(c)>127 for c in letters)
    return 'cjk' if cjk/max(1,len(letters))>.25 else 'other' if nonlatin/max(1,len(letters))>.25 else 'latin'


def fetch_encoder(root):
    from huggingface_hub import hf_hub_download
    import shutil
    root = Path(root); root.mkdir(parents=True,exist_ok=True)
    files = {}
    for filename in (ONNX,'tokenizer.json','README.md'):
        cached = hf_hub_download(MODEL,filename,revision=REVISION)
        target = root/Path(filename).name
        shutil.copyfile(cached,target); files[target.name] = file_sha(target)
    (root/'manifest.json').write_text(json.dumps(dict(model=MODEL,revision=REVISION,files=files,viewVersion=VIEW_VERSION),indent=2))


class Encoder:
    def __init__(self,root):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        root = Path(root); manifest = json.loads((root/'manifest.json').read_text())
        if manifest['revision'] != REVISION or manifest['viewVersion'] != VIEW_VERSION:
            raise ValueError('Wrong semantic encoder revision/views')
        for filename,digest in manifest['files'].items():
            if file_sha(root/filename) != digest: raise ValueError('Encoder file changed: '+filename)
        self.tokenizer = Tokenizer.from_file(str(root/'tokenizer.json'))
        self.tokenizer.enable_truncation(max_length=128,strategy='longest_first')
        pad = self.tokenizer.token_to_id('<pad>')
        if pad is None: raise ValueError('Expected original multilingual pad token')
        self.tokenizer.enable_padding(pad_id=pad,pad_token='<pad>')
        opts = ort.SessionOptions(); opts.intra_op_num_threads=4; opts.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(root/Path(ONNX).name),sess_options=opts,providers=['CPUExecutionProvider'])
        self.names={i.name for i in self.session.get_inputs()}

    def encode(self,rows,batch_size=32):
        values=[]
        for start in range(0,len(rows),batch_size):
            encoded=self.tokenizer.encode_batch([windows(r) for r in rows[start:start+batch_size]])
            inputs=dict(input_ids=np.asarray([e.ids for e in encoded],dtype=np.int64),
                        attention_mask=np.asarray([e.attention_mask for e in encoded],dtype=np.int64),
                        token_type_ids=np.asarray([e.type_ids for e in encoded],dtype=np.int64))
            output=self.session.run(None,{k:v for k,v in inputs.items() if k in self.names})[0]
            if output.ndim!=3 or output.shape[2]!=384: raise ValueError('Expected token embeddings, not hidden classifier output')
            mask=inputs['attention_mask'][...,None].astype(np.float32)
            pooled=(output*mask).sum(1)/np.maximum(mask.sum(1),1)
            pooled=pooled/np.maximum(np.linalg.norm(pooled,axis=1,keepdims=True),1e-12)
            if not np.isfinite(pooled).all(): raise ValueError('Nonfinite semantic embedding')
            values.append(pooled.astype(np.float32))
            if start%4096==0: print('Semantic embeddings',start,'/',len(rows),flush=True)
        return np.concatenate(values) if values else np.empty((0,384),dtype=np.float32)


def neighbours(embeddings,anchors,labels,batch_size=256):
    if embeddings.shape[1]!=anchors.shape[1]: raise ValueError('Embedding dimensions differ')
    labels=np.asarray(labels); results=[]
    if set(labels)!={0,1}: raise ValueError('Both TRAIN classes required for neighbours')
    for start in range(0,len(embeddings),batch_size):
        similarities=embeddings[start:start+batch_size]@anchors.T
        ham=np.max(similarities[:,labels==0],axis=1)
        spam=np.max(similarities[:,labels==1],axis=1)
        k=min(7,len(anchors)); indices=np.argpartition(similarities,-k,axis=1)[:,-k:]
        local=labels[indices].mean(1)
        results.append(np.column_stack([ham,spam,spam-ham,local]).astype(np.float32))
    return np.concatenate(results) if results else np.empty((0,4),dtype=np.float32)
