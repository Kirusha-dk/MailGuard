"""Real MIME CLI/API parity, including rescued HAM and excluded/conflicting mail."""
import argparse
import csv
import json
import subprocess
import sys
from email.message import EmailMessage
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np

import v77_mail as core
import v81_mail as runtime
from train_v80_en import read_rows
from train_v81_fp import paired
from v81_features import cached_fast, cached_neural
from v80_encoder import file_sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', type=Path, default=Path('reports/v81'))
    p.add_argument('--output', type=Path, default=Path('.cache/v81/cli-smoke'))
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    model = a.model_dir / 'model.joblib'; bundle = joblib.load(model)
    runtime.attach(bundle, SimpleNamespace(model=model))
    rows = read_rows(Path('.cache/v79/frozen/cal.jsonl.gz')) + read_rows(Path('.cache/v80/extra/cal.jsonl.gz'))
    old = bundle['_baseline']; policy = bundle['policy']; cache = Path('.cache/v81/features')
    fast = cached_fast(a.model_dir / 'baseline/model.joblib', rows, cache / 'cal-fast.npy')
    original = runtime.base_scores(fast) >= policy['baseline_cutoff']
    if policy['enabled']:
        f, e = cached_neural(old, rows, fast, original, cache / 'cal-embedding.npy')
        scores = runtime.candidate_scores(bundle, policy['candidate'], f, e,
                                          runtime.lexical_features([r['text'] for r in rows]))
        proposed = runtime.apply_policy(fast, scores, policy)
    else:
        proposed = original.copy()
    chosen = []
    for label in (0, 1):
        indices = [i for i, r in enumerate(rows) if r['y'] == label]
        indices.sort(key=lambda i: (not (original[i] and not proposed[i]),
                                    core.digest('v81-cli:' + rows[i]['identity'])))
        chosen += [rows[i] for i in indices[:30]]
    for name in ('ham', 'spam'):
        (a.output / name).mkdir()
    def write(name, text, label):
        message = EmailMessage(); message.set_content(text)
        (a.output / ('spam' if label else 'ham') / (name + '.eml')).write_bytes(message.as_bytes())
    for i, row in enumerate(chosen):
        write(str(i).zfill(3), row['text'], row['y'])
    for i, text in enumerate([
        'Bonjour, notre prochaine réunion se tiendra demain matin. Merci de confirmer votre présence et de préparer les documents nécessaires pour cette discussion.',
        'Guten Tag, die nächste Besprechung findet morgen statt. Bitte bestätigen Sie Ihre Teilnahme und bringen Sie die notwendigen Unterlagen für diese Diskussion mit.',
        'Добрый день, следующая встреча состоится завтра утром. Пожалуйста, подтвердите своё участие и подготовьте необходимые документы для обсуждения.']):
        write('foreign-' + str(i), text, 0)
    write('canonical-duplicate', chosen[0]['text'] + '!!!', 0)
    # A separate identity with conflicting labels must remove both copies.
    conflict = next(r for r in rows if r['y'] == 0 and r['identity'] not in {x['identity'] for x in chosen})
    write('conflict-ham', conflict['text'], 0); write('conflict-spam', conflict['text'], 1)
    reports, predictions = {}, {}
    for profile in runtime.PROFILES:
        out = a.output / profile
        result = subprocess.run([sys.executable, str(Path(__file__).parent / 'v81_mail.py'),
                                 '--model', str(model), '--ham', str(a.output / 'ham'),
                                 '--spam', str(a.output / 'spam'), '--profile', profile, '--output', str(out)],
                                check=True, capture_output=True, text=True)
        summary = json.loads((out / 'summary.json').read_text())
        records = list(csv.DictReader((out / 'predictions.csv').open()))
        api = runtime.predict(bundle, [r['text'] for r in chosen], profile)
        expected = {r['identity']: int(v['spam']) for r, v in zip(chosen, api)}
        actual = {r['identity']: int(r['predicted']) for r in records}
        if actual != expected or summary['metrics']['n'] != 60 or summary['excluded']['unsupported_or_short'] != 3 or summary['excluded']['conflicting_identities'] != 1:
            raise AssertionError('CLI/API/exclusion parity failed')
        y = np.array([r['y'] for r in chosen]); pred = np.array([v['spam'] for v in api])
        if summary['metrics']['tp'] != int((pred & (y == 1)).sum()) or summary['metrics']['fp'] != int((pred & (y == 0)).sum()):
            raise AssertionError('CLI metrics disagree with final decisions')
        reports[profile] = summary; predictions[profile] = pred
    probes = chosen[:4] + chosen[30:34]
    batched = runtime.predict(bundle, [r['text'] for r in probes], 'fp-rescue')
    single = [runtime.predict(bundle, [r['text']], 'fp-rescue')[0] for r in probes]
    difference = max(abs(a['score'] - b['score']) for a, b in zip(single, batched))
    if any(a['spam'] != b['spam'] for a, b in zip(single, batched)) or difference > 1e-7:
        raise AssertionError('Single/batch decisions disagree')
    report = dict(model_sha256=file_sha(model), profiles=reports, single_batch_max_score_difference=difference,
                  cli_api_decisions_equal=True, final_score_and_metrics_consistent=True,
                  smoke_is_not_a_generalization_test=True,
                  paired=paired(predictions['v80-recall-98'], predictions['fp-rescue'], chosen))
    (a.model_dir / 'cli-smoke.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
