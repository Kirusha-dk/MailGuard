"""Read-only probe of a frozen ES adapter on a small augmented Korean preview.

This does not train a Korean model or establish native-email performance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

SOURCE_URL = ('https://huggingface.co/datasets/joohans/korean-phishing-email/resolve/'
              'b8af3cc78ee0c6b475819169053304842f952587/data/korean_phishing_samples.jsonl')
SOURCE_SHA = 'dde0e7bfb84b4b972e6490ccf6cba5e11b64ed0fa35fd6640985025185722f15'
CHECKPOINT_SHA = 'b3693d61f8154b3085082d93afddd3654754efae0bf38ffce242cf4e78a0be70'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import torch
    import fasttext
    from transformers import AutoModel, AutoTokenizer
    from fetch_v83_inputs import FILES, LID_SHA, MODEL, REVISION, fetch
    from train_v83_language import CONFIG, frozen_parameter_hash, metric
    from v77_mail import clean
    from v82_neural import NeuralMail, load_trainable, text_tokenizer, trainable_state

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--backbone', type=Path, required=True)
    p.add_argument('--lid', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    source = args.out / 'korean-preview.jsonl'
    fetch(SOURCE_URL, source, SOURCE_SHA, limit=100_000)
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    if (len(rows) != 30 or any(type(r.get('label')) is not int or r['label'] not in (0,1)
                              or not isinstance(r.get('text'),str) for r in rows)
            or sum(r['label'] for r in rows) != 15):
        raise ValueError('Preview schema or original label counts drifted')
    if digest(args.checkpoint) != CHECKPOINT_SHA or digest(args.lid) != LID_SHA:
        raise ValueError('Pinned checkpoint or language detector mismatch')
    manifest = json.loads((args.backbone/'manifest.json').read_text())
    if manifest != dict(model=MODEL,revision=REVISION,files=FILES):
        raise ValueError('Multilingual backbone manifest differs')
    for name, expected in FILES.items():
        if digest(args.backbone/name) != expected:
            raise ValueError('Backbone file checksum mismatch')
    torch.set_num_threads(3);torch.manual_seed(CONFIG['seed'])
    torch.use_deterministic_algorithms(True)
    c = torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if (c['metadata']['language'] != 'es' or c['metadata']['config'] != CONFIG
            or c['progress']['step'] != 300 or c['metadata']['backbone'] != manifest):
        raise ValueError('Expected frozen Spanish step300 checkpoint')
    model = NeuralMail(AutoModel.from_pretrained(args.backbone,local_files_only=True,
                          use_safetensors=True,attn_implementation='eager'), CONFIG)
    load_trainable(model,c['adapter']);model.eval()
    before = trainable_state(model);frozen_before = frozen_parameter_hash(model)
    tokenizer = AutoTokenizer.from_pretrained(args.backbone,local_files_only=True)
    lid = fasttext.load_model(str(args.lid))
    texts = [clean('',r['text']) for r in rows]
    identities = [hashlib.sha256(t.encode()).hexdigest() for t in texts]
    if len(set(identities)) != len(identities):
        raise ValueError('Normalized duplicates require a new predeclared probe')
    language_results = []
    for text in texts:
        view = re.sub(r'\b(?:numbertoken|urltoken|emailtoken)\b',' ',text)
        pred = lid.f.predict(' '.join(view[:4000].split())+'\n',1,0.,'strict')[0]
        language_results.append(dict(language=pred[1].removeprefix('__label__'),confidence=pred[0]))
    scores = []
    with torch.no_grad():
        for start in range(0,len(texts),8):
            scores.extend(model(**text_tokenizer(tokenizer,texts[start:start+8],CONFIG['max_length'])).sigmoid().tolist())
    after = trainable_state(model)
    if (any(not torch.equal(before[k],after[k]) for k in before)
            or frozen_before != frozen_parameter_hash(model) or digest(args.checkpoint) != CHECKPOINT_SHA):
        raise ValueError('Read-only probe modified model or checkpoint')
    supported = [i for i,x in enumerate(language_results) if x['language']=='ko' and x['confidence']>=.8]
    selected = [dict(y=rows[i]['label']) for i in supported]
    metrics = metric(selected,[scores[i] for i in supported],.5) if supported else None
    report = dict(experiment='frozen_spanish_adapter_on_augmented_korean_preview',
        date='2026-11-04',source=dict(repository='joohans/korean-phishing-email',
        revision='b8af3cc78ee0c6b475819169053304842f952587',url=SOURCE_URL,sha256=SOURCE_SHA,
        license='Apache-2.0',rows=30,original_positive_labels=15,original_negative_labels=15,
        provenance='Publisher describes Korean augmentation; native-email origin not established',
        text_only_no_mime_or_separate_subject=True),
        model=dict(trained_language='es',campaign='v83-es-20261102',step=300,
        checkpoint_sha256=CHECKPOINT_SHA,backbone_revision=REVISION),
        korean_language_gate=dict(method='pinned fastText lid.176.ftz',minimum_confidence=.8,
        eligible=len(supported),abstentions=30-len(supported)),
        preview_at_locked_threshold_0_5=metrics,optimizer_steps_executed=0,
        checkpoint_unchanged=True,adapters_unchanged=True,backbone_unchanged=True,
        training_ready=False,korean_model_trained=False,native_email_validation=False,
        is_independent_native_email_test=False,threshold_tuned_on_preview=False,
        es_train_select_test_opened=False,customer_mail_used=False,accepted_for_production=False,
        limitations=['Tiny augmented preview, not representative Korean email',
        'Frozen Spanish adapters: this measures a diagnostic cross-language probe, not Korean fine-tuning',
        'Not enough labelled native mail for separate TRAIN/SELECT/TEST; no training gate bypass'])
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.out/'scores.json').write_text(json.dumps([dict(identity=identities[i],y=r['label'],
        score=scores[i],**language_results[i]) for i,r in enumerate(rows)],indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
