"""Pinned English MiniLM; portable signed 7-bit weights, independent inference."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
VIEW = 'english-minilm-s7-independent-head1200-tail600-pair128-v1'


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def window(text):
    return text if len(text) <= 1800 else (text[:1200], text[-600:])


class Encoder:
    def __init__(self, root, threads=1, workers=4):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        root = Path(root)
        manifest = json.loads((root / 'manifest.json').read_text())
        self.manifest_sha = file_sha(root / 'manifest.json')
        if manifest['revision'] != REVISION or manifest['view'] != VIEW:
            raise ValueError('Unexpected neural encoder or text view')
        for name, sha in manifest['files'].items():
            if file_sha(root / name) != sha:
                raise ValueError('Encoder checksum mismatch: ' + name)
        self.tokenizer = Tokenizer.from_file(str(root / 'tokenizer.json'))
        self.tokenizer.enable_truncation(max_length=128, strategy='longest_first')
        pad = self.tokenizer.token_to_id('[PAD]')
        if pad != 0:
            raise ValueError('Unexpected English padding token')
        self.tokenizer.enable_padding(length=128, pad_id=pad, pad_token='[PAD]')
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(root / 'model.onnx'), options,
                                           providers=['CPUExecutionProvider'])
        self.names = {item.name for item in self.session.get_inputs()}
        self.workers = workers

    def _one(self, token):
        features = dict(input_ids=np.asarray([token.ids], dtype=np.int64),
                        attention_mask=np.asarray([token.attention_mask], dtype=np.int64),
                        token_type_ids=np.asarray([token.type_ids], dtype=np.int64))
        output = self.session.run(None, {k: v for k, v in features.items() if k in self.names})[0]
        if output.ndim != 3 or output.shape[2] != 384:
            raise ValueError('Expected 384-dimensional token embeddings')
        mask = features['attention_mask'][..., None].astype(np.float32)
        pooled = (output * mask).sum(1) / np.maximum(mask.sum(1), 1)
        pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
        if not np.isfinite(pooled).all():
            raise ValueError('Nonfinite neural features')
        return pooled[0].astype(np.float32)

    def encode(self, texts, batch_size=1, progress=False):
        # Dynamic activation quantization is batch-dependent. Never let other
        # emails alter a message's neural features or its threshold decision.
        if batch_size != 1:
            raise ValueError('Independent neural inference requires batch_size=1')
        values = []
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            for start in range(0, len(texts), 64):
                tokens = self.tokenizer.encode_batch([window(t) for t in texts[start:start + 64]])
                # Concurrent independent one-message runs, never a quantized
                # multi-message tensor. executor.map preserves original order.
                values.extend(executor.map(self._one, tokens))
                if progress and start % 4096 == 0:
                    print('Encoded', start, '/', len(texts), flush=True)
        return np.asarray(values, dtype=np.float32).reshape(-1, 384)
