"""Bounded English replay, with explicit v82 migration and exact resumability."""
import datetime as dt
import hashlib
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
from torch import nn

from v82_neural import CONFIG as PARENT_CONFIG, load_trainable
from v84_replay_data import PARENT_SHA, PARENT_STEP, sha

START = '2026-11-05'
END = '2027-05-04'
CAMPAIGN = 'v84-en-replay-20261105'
CONFIG = dict(PARENT_CONFIG, version='v84-en-replay-r8-v1', seed=84,
              daily_rows=50000, max_epochs=3, max_row_exposures=3,
              error_multiplier=2., known_error_multiplier=1.5, patience=5)
FIXED_THRESHOLD = 0.04897100105881691


def check_request(request, today=None):
    today = today or dt.datetime.now(ZoneInfo('Europe/Moscow')).date()
    day = dt.date.fromisoformat(request['session_id'])
    if (request['campaign'] != CAMPAIGN or request['start_date'] != START or
            request['end_date'] != END or not dt.date.fromisoformat(START) <= day <=
            min(today, dt.date.fromisoformat(END)) or today > dt.date.fromisoformat(END)):
        raise ValueError('Wrong campaign, expired window or future session')
    if request['budget_rows'] != 50000 or request['budget_steps'] != 3125:
        raise ValueError('Original daily target must remain50000 rows/3125 steps')
    if request['expected_min_step'] < PARENT_STEP:
        raise ValueError('Rollback floor must include inherited8500 steps')


def new_progress(rows, parent, parent_lr):
    return dict(step=PARENT_STEP, epoch=0, cursor=0, order=None,
        exposures=torch.tensor([r['parent_exposures'] for r in rows], dtype=torch.uint8),
        active_session=None, active_budget_rows=None, active_budget_steps=None,
        active_start=None, active_rows=0, active_seen=[], active_loss_sum=0.,
        active_wrong_rows=0, active_ham_rows=0, active_spam_rows=0,
        phase_rows=0, completed_sessions=[], stalled=0, stopped=False,
        best_fp=parent['progress']['last_report']['development']['fp'],
        best_loss=parent['progress']['last_report']['development']['log_loss'],
        parent_learning_rate=parent_lr, last_report=None)


def migrate(parent_path, model, optimizer, rows, metadata):
    if sha(parent_path) != PARENT_SHA:
        raise ValueError('Locked parent checkpoint missing or different')
    parent = torch.load(parent_path, map_location='cpu', weights_only=True)
    if (parent['progress']['step'] != PARENT_STEP or parent['progress']['active_session'] is not None or
            parent['metadata']['train'] != metadata['corpus']['original_v82_train_fingerprint'] or
            parent['metadata']['select'] != metadata['corpus']['original_v82_select_fingerprint'] or
            parent['metadata']['config'] != PARENT_CONFIG or
            parent['metadata']['backbone'] != metadata['backbone'] or
            parent['metadata']['torch_version'] != metadata['torch_version'] or
            parent['metadata']['transformers_version'] != metadata['transformers_version']):
        raise ValueError('Parent state, environment or provenance mismatch')
    load_trainable(model, parent['adapter']); optimizer.load_state_dict(parent['optimizer'])
    lr = optimizer.param_groups[0]['lr']
    # A new corpus/objective needs an explicit new scheduler, while preserving Adam.
    for group in optimizer.param_groups:
        group.pop('initial_lr', None)
    torch.set_rng_state(parent['rng'])
    return new_progress(rows, parent, lr)


def begin_session(p, request):
    if request['session_id'] in p['completed_sessions'] or p['stopped']:
        return False
    maximum = (dt.date.fromisoformat(END)-dt.date.fromisoformat(START)).days+1
    if len(p['completed_sessions']) >= maximum:
        raise ValueError('Campaign date cap reached')
    if p['active_session'] is not None:
        if (request['session_id'] != p['active_session'] or
                request['budget_rows'] != p['active_budget_rows'] or
                request['budget_steps'] != p['active_budget_steps']):
            raise ValueError('Resume original session, row target and step budget')
        return True
    p.update(active_session=request['session_id'], active_budget_rows=request['budget_rows'],
             active_budget_steps=request['budget_steps'], active_start=p['step'], active_rows=0,
             active_seen=[], active_loss_sum=0., active_wrong_rows=0, active_ham_rows=0, active_spam_rows=0)
    return True


def make_order(rows, p):
    eligible = [i for i in range(len(rows)) if p['exposures'][i].item() < 3]
    # Known errors are encountered early, but each row is used at most once per
    # epoch and never more than3 times including its parent v82 training history.
    eligible.sort(key=lambda i: (not rows[i]['hard'], hashlib.sha256(
        f"84:{p['epoch']}:{rows[i]['identity']}".encode()).hexdigest()))
    return torch.tensor(eligible, dtype=torch.int64)


def run_steps(model, tokenizer, rows, weights, optimizer, scheduler, p, tokenize,
              invocation_limit=3125, save=None):
    if p['active_session'] is None:
        raise ValueError('No active session')
    seen = set(p['active_seen']); initial = p['step']; model.train()
    while (p['active_rows'] < p['active_budget_rows'] and
           p['step']-p['active_start'] < p['active_budget_steps'] and
           p['step']-initial < invocation_limit and p['epoch'] < 3):
        ix = []
        size = min(16, p['active_budget_rows']-p['active_rows'])
        while len(ix) < size and p['epoch'] < 3:
            if p['order'] is None:
                p['order'] = make_order(rows, p); p['cursor'] = 0
            if not len(p['order']):
                p['epoch'] = 3; break
            if p['cursor'] == len(p['order']):
                p['epoch'] += 1; p['order'] = None; p['cursor'] = 0; continue
            i = int(p['order'][p['cursor']]); p['cursor'] += 1
            if i not in seen and p['exposures'][i].item() < 3:
                ix.append(i); seen.add(i)
        if not ix:
            break
        batch = [rows[i] for i in ix]
        inputs = tokenize(tokenizer, [r['text'] for r in batch], CONFIG['max_length'])
        y = torch.tensor([r['y'] for r in batch], dtype=torch.float32)
        base = torch.tensor([weights[i] for i in ix], dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        logits = model(**inputs)
        wrong = ((logits.detach().sigmoid() >= FIXED_THRESHOLD) != y.bool())
        known = torch.tensor([r['hard'] for r in batch], dtype=torch.bool)
        multiplier = torch.where(wrong, CONFIG['error_multiplier'], 1.)
        multiplier *= torch.where(known, CONFIG['known_error_multiplier'], 1.)
        loss = (nn.functional.binary_cross_entropy_with_logits(logits, y, reduction='none') * base * multiplier).mean()
        if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
        loss.backward()
        nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad], 1., error_if_nonfinite=True)
        optimizer.step(); scheduler.step()
        p['step'] += 1; p['active_rows'] += len(ix); p['phase_rows'] += len(ix)
        p['active_seen'].extend(ix); p['exposures'][ix] += 1
        p['active_loss_sum'] += float(loss.detach()) * len(ix)
        p['active_wrong_rows'] += int(wrong.sum()); p['active_spam_rows'] += int(y.sum())
        p['active_ham_rows'] += len(ix)-int(y.sum())
        if p['order'] is not None and p['cursor'] == len(p['order']):
            p['epoch'] += 1; p['order'] = None; p['cursor'] = 0
        if save and (p['step']-PARENT_STEP) % 25 == 0:
            save()
            print(dict(step=p['step'], session_rows=p['active_rows'], target_rows=p['active_budget_rows']), flush=True)
    if len(p['active_seen']) != len(set(p['active_seen'])) or p['exposures'].max().item() > 3:
        raise ValueError('Exposure or within-session uniqueness violation')
    return (p['active_rows'] >= p['active_budget_rows'] or p['epoch'] >= 3 or
            p['step']-p['active_start'] >= p['active_budget_steps'])


def confusion(labels, scores, threshold):
    import numpy as np
    y = np.asarray(labels); s = np.asarray(scores); detected = s >= threshold
    ham = int((y == 0).sum()); spam = int((y == 1).sum())
    tp = int((detected & (y == 1)).sum()); fp = int((detected & (y == 0)).sum())
    return dict(n=len(y), spam=spam, ham=ham, tp=tp, fp=fp, fn=spam-tp, tn=ham-fp,
                threshold=threshold, recall=tp/spam if spam else None,
                fpr=fp/ham if ham else None, fp_per_million_ham=fp/ham*1e6 if ham else None)


def development(model, tokenizer, rows, tokenize):
    import numpy as np
    model.eval(); values = []
    with torch.no_grad():
        for start in range(0, len(rows), 32):
            values.extend(model(**tokenize(tokenizer, [r['text'] for r in rows[start:start+32]],192)).sigmoid().tolist())
    y = np.array([r['y'] for r in rows]); s = np.asarray(values)
    if not np.isfinite(s).all(): raise ValueError('Nonfinite evaluation')
    spam = np.sort(s[y == 1])
    cutoff = float(spam[int(math.floor(.02*len(spam)))])
    result = confusion(y, s, cutoff)
    clipped = np.clip(s, 1e-7, 1-1e-7)
    result.update(log_loss=float(-(y*np.log(clipped)+(1-y)*np.log(1-clipped)).mean()),
        fixed_parent_threshold=confusion(y, s, FIXED_THRESHOLD),
        is_development=True, is_independent_test=False,
        threshold_selected_on_same_development=True,
        by_source={source: confusion([r['y'] for r in rows if r['source']==source],
                     [v for r,v in zip(rows, values) if r['source']==source], cutoff)
                   for source in sorted({r['source'] for r in rows})})
    return result
