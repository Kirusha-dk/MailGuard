"""Deterministic stopped-experiment package: weights/state/metadata, never mail.

This archival command imports no training or neural library and performs no
optimizer steps. Metadata has been independently checked against the checkpoint
by the daily controller before publication.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def sha(value):
    return hashlib.sha256(value).hexdigest()


def package(request, state, output):
    info = dict(request); expected = info.pop('expected_package_sha256', None)
    lang = info['language']; campaign = info['campaign']
    if lang not in ('es', 'tr') or campaign != f'v83-{lang}-'+('20261102' if lang=='es' else '20261104'):
        raise ValueError('Unsupported experiment')
    report = json.loads((state/'report.json').read_text())
    if (report['language']!=lang or report['campaign']!=campaign or not report['stopped'] or
            report['global_optimizer_steps']!=info['step'] or report['test_or_cal_opened'] or
            report['accepted_for_production'] or report['customer_mail_used']):
        raise ValueError('Expected a stopped, verified experiment')
    if sha((state/'checkpoint.pt').read_bytes())!=info['checkpoint_sha256'] or report['checkpoint_sha256']!=info['checkpoint_sha256']:
        raise ValueError('Checkpoint differs from verified state')
    meta = json.loads(Path(info['metadata_path']).read_text())
    frozen = json.loads(Path('research/v83',lang+'-frozen.json').read_text())
    verification = json.loads(Path(info['verification_path']).read_text())
    if (meta['language']!=lang or meta['campaign']!=campaign or
            verification['checkpoint_sha256']!=info['checkpoint_sha256'] or
            not verification['select_metrics_independently_recomputed_without_gradients'] or
            verification['verification_optimizer_steps']!=0):
        raise ValueError('Verification/metadata mismatch')
    # Explicit allowlist prevents accidentally archiving corpora or raw messages.
    blobs = {name:(state/name).read_bytes() for name in
             ('checkpoint.pt','best-adapter.safetensors','best-development.json','report.json')}
    blobs.update({'metadata.json':json.dumps(meta,indent=2).encode()+b'\n',
        'backbone-manifest.json':json.dumps(meta['backbone'],indent=2).encode()+b'\n',
        'source-frozen-manifest.json':json.dumps(frozen,indent=2).encode()+b'\n',
        'verification.json':json.dumps(verification,indent=2).encode()+b'\n',
        'archive-request.json':json.dumps(info,indent=2).encode()+b'\n',
        'package_v83_experiment.py':Path(__file__).read_bytes()})
    metric = report['development']['at_fixed_0_5']
    blobs['README.md'] = f'''# MailGuard {campaign}: stopped experimental package

Not accepted for production. This is an experimental {'Spanish phishing' if lang=='es' else 'Turkish spam'} classifier.
Stopped at step{info['step']}: {report['stop_reason']}; epoch{report['epoch']}.
Authorized window: {meta['start_date']} through {meta['end_date']}, inclusive.
Do not restart or automatically extend training after this stop.

Source run: https://github.com/Kirusha-dk/MailGuard/actions/runs/{info['source_run_id']}
Training commit: {info['training_commit']}.
SELECT development, fixed threshold0.5: TP{metric['tp']} FP{metric['fp']} FN{metric['fn']} TN{metric['tn']}.
HAM denominator{metric['ham']}; recall{metric['recall']:.8%}, FPR{metric['fpr']:.8%}.
TEST was not opened for gradients, selection or independent verification.
These development observations do not establish general spam accuracy or10ppm.

checkpoint.pt preserves LoRA/head, Adam, scheduler, RNG and data cursor.
Best adapter is selected by development log loss, not a guarantee of lower FP.
Backbone weights are not bundled; exact revision and per-file SHA256 checks
are in backbone-manifest.json. Restore only the pinned CPU environment and
trusted source code. This stopped state is for audit/inference, not more training.
Verify SHA256SUMS.json before loading. No raw emails/customer mail are included.
'''.encode()
    attribution = '''# Source attribution and limits

Backbone: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2,
revision e8f8c211226b894fcb81acc59f3b34ba3efd5f42.
Publisher files/terms: https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/tree/e8f8c211226b894fcb81acc59f3b34ba3efd5f42 .
No pretrained backbone weights or dataset texts are redistributed here.

'''
    if lang=='es':
        attribution += '''SpaPhish authors, version5: https://data.mendeley.com/datasets/hz2d6gz7pc/5 .
License CC BY4.0: https://creativecommons.org/licenses/by/4.0/ .
Only subject/body were normalized, deduplicated, language-filtered and split;
expert annotations were not used as features and labels were not invented.
'''
    else:
        attribution += '''UCI Turkish Spam V01: https://doi.org/10.24432/C5WG7F , CC BY4.0.
Simsek and Aydemir Turkish Mail Dataset Normal/Spam version1:
https://www.kaggle.com/datasets/emrahaydemr/turkish-mail-dataset-normalspam .
Publisher requires citation: https://doi.org/10.55195/jscai.1104694 .
The second source supplies no standard license; do not call all data CC BY4.0.
CSV repairs, excluded footer, source hashes and unchanged label mappings
are preserved in source-frozen-manifest.json. No raw mail is redistributed.
'''
    blobs['ATTRIBUTION.md'] = attribution.encode()
    blobs['SHA256SUMS.json'] = (json.dumps({k:sha(v) for k,v in sorted(blobs.items())},indent=2)+'\n').encode()
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'w') as z:
        for name,value in sorted(blobs.items()):
            entry = zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0))
            # ZIP_STORED avoids differences between zlib versions on workers.
            entry.compress_type=zipfile.ZIP_STORED;entry.external_attr=0o100644<<16
            z.writestr(entry,value)
    digest = sha(output.read_bytes())
    if expected is not None and digest!=expected:
        output.unlink();raise ValueError('Deterministic package differs from reviewed SHA256')
    output.with_suffix(output.suffix+'.sha256').write_text(digest+'  '+output.name+'\n')
    output.with_suffix('.notes.md').write_bytes(blobs['README.md']+b'\n'+blobs['ATTRIBUTION.md'])
    return digest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--request',type=Path,required=True)
    p.add_argument('--state',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();print(package(json.loads(args.request.read_text()),args.state,args.output))
