"""Train internal MiniLM adapters with exactly resumable bounded sessions."""
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn

MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
CONFIG = dict(version='v82-lora-minilm-r8-v1', model=MODEL, revision=REVISION,
              rank=8, alpha=16, max_length=192, seed=82, batch_size=16,
              learning_rate=.0002, weight_decay=.01, max_epochs=3, ham_cost=2.)


def fingerprint(rows):
    digest = hashlib.sha256()
    for row in rows:
        data = json.dumps({k: row[k] for k in ('identity', 'family', 'source', 'y', 'text')},
                          ensure_ascii=False, sort_keys=True).encode()
        digest.update(len(data).to_bytes(8, 'big')); digest.update(data)
    return digest.hexdigest()


class LoRALinear(nn.Module):
    def __init__(self, original, rank, alpha):
        super().__init__()
        self.original = original
        for parameter in original.parameters():
            parameter.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, original.in_features))
        self.b = nn.Parameter(torch.zeros(original.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x):
        return self.original(x) + (x @ self.a.T @ self.b.T) * self.scale


class NeuralMail(nn.Module):
    def __init__(self, backbone, config=CONFIG):
        super().__init__()
        self.backbone = backbone
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        for layer in backbone.encoder.layer:
            attention = layer.attention.self
            for name in ('query', 'value'):
                setattr(attention, name, LoRALinear(getattr(attention, name), config['rank'], config['alpha']))
        self.dropout = nn.Dropout(.1)
        self.head = nn.Linear(backbone.config.hidden_size, 1)

    def forward(self, **inputs):
        hidden = self.backbone(**inputs).last_hidden_state
        mask = inputs['attention_mask'].unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        return self.head(self.dropout(pooled)).squeeze(-1)


def trainable_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.named_parameters() if value.requires_grad}


def load_trainable(model, state):
    parameters = {k: v for k, v in model.named_parameters() if v.requires_grad}
    if parameters.keys() != state.keys() or any(parameters[k].shape != state[k].shape for k in state):
        raise ValueError('Adapter architecture mismatch')
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value)


def optimizer_and_scheduler(model, total_steps, config=CONFIG):
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=config['learning_rate'], weight_decay=config['weight_decay'])
    warmup = min(100, max(1, total_steps // 10))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda step: min((step + 1) / warmup, max(0., (total_steps - step) / max(1, total_steps - warmup))))
    return optimizer, scheduler


def new_progress():
    return dict(step=0, epoch=0, cursor=0, completed_sessions=[], active_session=None,
                active_target=None, best_fp=None, best_loss=None, stalled=0, stopped=False)


def save_checkpoint(path, model, optimizer, scheduler, progress, metadata):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix('.partial')
    torch.save(dict(adapter=trainable_state(model), optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(), progress=progress, metadata=metadata,
                    rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), pending)
    pending.replace(path)


def restore_checkpoint(path, model, optimizer, scheduler, metadata):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint['metadata'] != metadata:
        raise ValueError('Checkpoint configuration, dataset, campaign or device mismatch')
    load_trainable(model, checkpoint['adapter'])
    optimizer.load_state_dict(checkpoint['optimizer']); scheduler.load_state_dict(checkpoint['scheduler'])
    torch.set_rng_state(checkpoint['rng'])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    return checkpoint['progress']


def begin_session(progress, session_id, budget):
    if session_id in progress['completed_sessions'] or progress['stopped']:
        return False
    if len(progress['completed_sessions']) >= 30:
        raise ValueError('Thirty completed sessions already reached')
    if progress['active_session'] not in (None, session_id):
        raise ValueError('Resume the unfinished session before requesting another')
    if progress['active_session'] is None:
        progress['active_session'] = session_id
        progress['active_target'] = progress['step'] + budget
    return True


def text_tokenizer(tokenizer, texts, max_length=192):
    heads = [text if len(text) <= 1800 else text[:1200] for text in texts]
    tails = ['' if len(text) <= 1800 else text[-600:] for text in texts]
    return tokenizer(heads, tails, padding=True, truncation='longest_first',
                     max_length=max_length, return_tensors='pt')


def run_steps(model, tokenizer, rows, weights, optimizer, scheduler, progress,
              config=CONFIG, save=None):
    device = next(model.parameters()).device
    losses = []; model.train(); order_epoch = None; order = None
    while progress['step'] < progress['active_target'] and progress['epoch'] < config['max_epochs']:
        if order_epoch != progress['epoch']:
            order_epoch = progress['epoch']
            order = sorted(range(len(rows)), key=lambda i: hashlib.sha256(
                f"{config['seed']}:{progress['epoch']}:{rows[i]['identity']}".encode()).hexdigest())
        indices = order[progress['cursor']:progress['cursor'] + config['batch_size']]
        batch = [rows[i] for i in indices]
        inputs = {k: v.to(device) for k, v in text_tokenizer(tokenizer,
                  [r['text'] for r in batch], config['max_length']).items()}
        labels = torch.tensor([r['y'] for r in batch], dtype=torch.float32, device=device)
        cost = torch.tensor([weights[i] for i in indices], dtype=torch.float32, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = (nn.functional.binary_cross_entropy_with_logits(model(**inputs), labels,
                                                               reduction='none') * cost).mean()
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        loss.backward()
        nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
        optimizer.step(); scheduler.step()
        losses.append(float(loss.detach()))
        progress['step'] += 1; progress['cursor'] += len(indices)
        if progress['cursor'] == len(rows):
            progress['epoch'] += 1; progress['cursor'] = 0
        if save and progress['step'] % 25 == 0:
            save()
            print(json.dumps(dict(step=progress['step'], epoch=progress['epoch'], loss=losses[-1])), flush=True)
    return losses


def validate(model, tokenizer, rows, config=CONFIG):
    import numpy as np
    device = next(model.parameters()).device
    scores = []; model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), 32):
            batch = rows[start:start + 32]
            inputs = {k: v.to(device) for k, v in text_tokenizer(tokenizer,
                      [r['text'] for r in batch], config['max_length']).items()}
            scores.extend(model(**inputs).sigmoid().cpu().tolist())
    labels = np.array([r['y'] for r in rows]); scores = np.array(scores)
    spam = np.sort(scores[labels == 1]); ham = scores[labels == 0]
    if len(spam) < 100 or not len(ham):
        raise ValueError('SELECT needs at least 100 spam and some HAM')
    cutoff = float(spam[int(math.floor(.02 * len(spam)))])
    tp = int((spam >= cutoff).sum()); fp = int((ham >= cutoff).sum())
    clipped = np.clip(scores, 1e-7, 1 - 1e-7)
    loss = float(-(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)).mean())
    return dict(n=len(rows), spam=len(spam), ham=len(ham), tp=tp, fp=fp, recall=tp / len(spam),
                fp_per_million_ham=fp / len(ham) * 1e6, threshold=cutoff, log_loss=loss,
                threshold_selected_on_same_development=True, is_independent_test=False,
                million_message_test_performed=False)
