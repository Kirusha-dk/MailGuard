import math
import torch
from v82_neural import CONFIG, text_tokenizer


def evaluate(model, tokenizer, rows, config=CONFIG):
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
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite development scores')
    spam = np.sort(scores[labels == 1]); ham = scores[labels == 0]
    if len(spam) < 100 or not len(ham):
        raise ValueError('SELECT needs at least 100 spam and some HAM')
    cutoff = float(spam[int(math.floor(.02 * len(spam)))])
    tp = int((spam >= cutoff).sum()); fp = int((ham >= cutoff).sum())
    clipped = np.clip(scores, 1e-7, 1 - 1e-7)
    loss = float(-(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)).mean())
    sources = np.array([r['source'] for r in rows]); by_source = {}
    for source in sorted(set(sources.tolist())):
        mask = sources == source; source_y = labels[mask]; detected = scores[mask] >= cutoff
        source_spam = int((source_y == 1).sum()); source_ham = int((source_y == 0).sum())
        source_tp = int((detected & (source_y == 1)).sum())
        source_fp = int((detected & (source_y == 0)).sum())
        by_source[source] = dict(n=int(mask.sum()), spam=source_spam, ham=source_ham,
                                 tp=source_tp, fp=source_fp,
                                 recall=source_tp / source_spam if source_spam else None,
                                 fp_per_million_ham=source_fp / source_ham * 1e6 if source_ham else None)
    result = dict(n=len(rows), spam=len(spam), ham=len(ham), tp=tp, fp=fp, recall=tp / len(spam),
                fp_per_million_ham=fp / len(ham) * 1e6, threshold=cutoff, log_loss=loss,
                by_source=by_source,
                threshold_selected_on_same_development=True, is_independent_test=False,
                million_message_test_performed=False)

    return result, scores.tolist()
