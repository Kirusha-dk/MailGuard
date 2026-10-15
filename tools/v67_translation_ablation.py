#!/usr/bin/env python3
"""v67 retrospective language-translation ablation on frozen v66 public data.

Never train or calibrate classifiers against v66, and never present this
already-scored cohort as another independent evaluation. Model contents and
thresholds are verified by SHA256 before any comparison.
"""
import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path

from evaluate_v66_firstuse import (
    V62_SHA, V64_SHA, V65_SHA, reconstruct, fingerprint,
    sha_file, wilson
)

SOURCE_AUDIT = Path('reports/v66-candidate-audit.json')
TRANSLATOR = 'Helsinki-NLP/opus-mt-tr-en'
TRANSLATOR_REVISION = 'a01ed04b0bd6624fb4f699e3a8d8880a7004e00f'
MAX_TRANSLATE_MESSAGES = 384  # label-blind, predeclared SHA256 sample
PROFILES = ('v64-conservative', 'v64-high-recall', 'v65-conservative-rescue')
SCENARIOS = ('original', 'translated_only', 'original_or_translation')


def stable_subset(rows, limit=MAX_TRANSLATE_MESSAGES):
    """Avoid pre-selecting positives, errors, particular classes, or outcomes."""
    if limit < 1:
        raise ValueError('Non-positive limit')
    subset = [r for r in rows if r['source'] == 'Turkish-stemmed']
    return sorted(subset,
                  key=lambda r: hashlib.sha256(
                      ('v67-label-blind-v1:' + r['identity']).encode('ascii')).digest()
                  )[:limit]


def tally(counter, label, positive):
    counter['tp' if (label == 1 and positive) else
            'fn' if label == 1 else
            'fp' if positive else 'tn'] += 1


def metrics(table):
    tp, fn, fp, tn = (table[x] for x in ('tp','fn','fp','tn'))
    return dict(n=tp+fn+fp+tn, spam=tp+fn, ham=fp+tn,
                tp=tp, fn=fn, fp=fp, tn=tn,
                recall=tp/(tp+fn) if tp+fn else None,
                fpr=fp/(fp+tn) if fp+tn else None,
                fprWilson95Upper=wilson(fp,fp+tn))


def OR(original, translated):
    return tuple(bool(a) or bool(b) for a,b in zip(original,translated))


def score_batch(bundle, expert, rows):
    from v64_runtime import score_rows
    from v65_domain_rescue import score_active, vectors
    v64=score_rows(bundle,rows)
    for i,r in enumerate(rows):
        r['v64ConservativeMargin']=float(v64['conservative'][i])
    # reuse v65's exactly frozen word/char vectorizers and specialist weights
    additional=score_active(vectors(),expert['models'],rows)
    decisions=[]
    for i,r in enumerate(rows):
        conservative=float(v64['conservative'][i])>=0
        high=float(v64['high-recall'][i])>=0
        assisted=r['script']!='cjk' and float(additional[i])>=expert['cutoff']
        decisions.append((conservative,high,conservative or assisted))
    return decisions


def translation_to_rows(tokenizer, translator, turkish, max_tokens=192):
    """Translate mail BODY only, not model markers or label fields.

    Conversion is for experiments; language-normalization may lose
    obfuscations, links, headers and other important anti-spam signals.
    """
    import torch
    from train_v47_phishing_aware import canonical_fields
    from train_v55_multilingual import views
    from v61_semantic import script
    rows=[];truncated=0;outputs_count=0
    for start in range(0,len(turkish),8):
        chunk=turkish[start:start+8]
        texts=[r['plain_body'] for r in chunk]
        # Explicitly count truncated inputs to interpret lost-content results.
        token_lengths=[len(x) for x in tokenizer(texts,add_special_tokens=True)['input_ids']]
        truncated+=sum(n>max_tokens for n in token_lengths)
        tokens=tokenizer(texts,return_tensors='pt',padding=True,
                         truncation=True,max_length=max_tokens)
        with torch.inference_mode():
            output=translator.generate(**tokens,max_new_tokens=160,num_beams=1,
                                       do_sample=False,use_cache=True)
        english=tokenizer.batch_decode(output,skip_special_tokens=True)
        if len(english)!=len(chunk):raise ValueError('Translation batch misaligned')
        for old,translated in zip(chunk,english):
            if not translated.strip():raise ValueError('Empty translation')
            content=canonical_fields('',translated)
            focus,context=views(content['ngram_text'])
            rows.append(dict(identity=old['identity'],family=old['family'],
                focus=focus,context=context,script=script(dict(context=context))))
            outputs_count+=1
        print('V67 translated public mail',outputs_count,'/',len(turkish),flush=True)
    return rows,truncated


def freeze(args,old_audit):
    """Replay the immutable v66 manifest: nothing about v66 outcome is tuned."""
    manifest=json.loads(Path(args.manifest).read_text())
    if manifest.get('stage')!='pre-inference-v66-fixed-first-use-cohort':
        raise ValueError('Wrong source manifest')
    if manifest['sourceAuditSha256']!=sha_file(SOURCE_AUDIT):
        raise ValueError('Source candidate audit changed')
    if manifest['previousV64TrainSha256']!=V64_SHA or manifest['previousV65ModelSha256']!=V65_SHA:
        raise ValueError('Frozen model identifiers changed')
    rows=reconstruct(args.prior,args.old_input,old_audit)
    if len(rows)!=manifest['count'] or fingerprint(rows)!=manifest['fingerprintSha256']:
        raise ValueError('Cohort changed since pre-inference v66 freeze')
    selected=stable_subset(rows)
    if len(selected)!=MAX_TRANSLATE_MESSAGES:
        raise ValueError('Too few Turkish messages for fixed sample')
    return selected,manifest


def recover_body(args, old_audit, chosen):
    """Reload source text only; never store it in GitHub output.

    Exact archive digests and identities are verified via the same v66
    filtering used for the original v66 first-use score.
    """
    import prepare_v66_candidate_audit as v66
    from prepare_v64_transfer import enable_equivalent_normalizers
    from train_v55_multilingual import fields
    enable_equivalent_normalizers()
    original=v66.row
    needed={r['identity'] for r in chosen}
    found={}
    def intercept(parsed,label,source,raw=None):
        record=original(parsed,label,source,raw)
        if record and source=='Turkish-stemmed' and record['identity'] in needed:
            body=fields(parsed['ngram_text'])[1]
            # The publisher's processed text has no MIME headers.
            found[record['identity']]=body
        return record
    v66.row=intercept
    try:
        # Do not repeat the 500k historical scan; freeze() already did it.
        v66.turkish()
    finally:
        v66.row=original
    if set(found)!=needed:raise ValueError('Unable to recover all frozen plain texts')
    for r in chosen:r['plain_body']=found[r['identity']]
    return chosen


def run(args):
    import joblib
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from v64_runtime import load_bundle
    from evaluate_v66_firstuse import V64_SHA, V62_SHA, V65_SHA

    old_audit=json.loads(SOURCE_AUDIT.read_text())
    chosen,manifest=freeze(args,old_audit)
    # Critically, this is a RETROSPECTIVE sample, selected by identity alone.
    selected_hash=hashlib.sha256('\n'.join(sorted(r['identity'] for r in chosen)).encode()).hexdigest()
    chosen=recover_body(args,old_audit,chosen)
    root=Path(args.model64);ext=Path(args.model65)/'model.joblib'
    for name,path,digest in (('v64',root/'model.joblib',V64_SHA),
                             ('v62',root/'v62/model.joblib',V62_SHA),
                             ('v65',ext,V65_SHA)):
        if sha_file(path)!=digest:raise ValueError('Changed frozen '+name+' model')
    bundle=load_bundle(root)
    expert=joblib.load(ext)
    if expert.get('version')!='v65-retrospective-adaptation' or not math.isfinite(expert['cutoff']):
        raise ValueError('Unexpected frozen specialist')
    print('V67 frozen v64/v62/v65 SHA checks passed',flush=True)
    torch.set_num_threads(2)
    tokenizer=AutoTokenizer.from_pretrained(TRANSLATOR,revision=TRANSLATOR_REVISION,use_fast=False)
    translator=AutoModelForSeq2SeqLM.from_pretrained(
        TRANSLATOR,revision=TRANSLATOR_REVISION,use_safetensors=True)
    translator.eval()
    translation_rows,truncated=translation_to_rows(tokenizer,translator,chosen)
    if len(translation_rows)!=len(chosen) or any(a['identity']!=b['identity'] for a,b in zip(translation_rows,chosen)):
        raise ValueError('Translated row alignment broken')
    by_group={mode:{p:Counter() for p in PROFILES} for mode in SCENARIOS}
    improvements={p:Counter() for p in PROFILES}
    for start in range(0,len(chosen),64):
        old=score_batch(bundle,expert,chosen[start:start+64])
        new=score_batch(bundle,expert,translation_rows[start:start+64])
        for r,op,np in zip(chosen[start:start+64],old,new):
            y=int(r['label'])
            if y not in (0,1):raise ValueError('Unexpected label')
            for i,profile in enumerate(PROFILES):
                for mode,pred in zip(SCENARIOS,(op[i],np[i],op[i] or np[i])):
                    tally(by_group[mode][profile],y,pred)
                if y:
                    if not op[i] and np[i]:improvements[profile]['rescuedFN']+=1
                    if op[i] and not np[i]:improvements[profile]['newFN_if_translated_only']+=1
                elif not op[i] and np[i]:
                    improvements[profile]['newFP_under_OR']+=1
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    table={mode:{p:metrics(t) for p,t in profiles.items()} for mode,profiles in by_group.items()}
    for mode,versions in table.items():
        for name,m in versions.items():
            if m['n']!=len(chosen):raise ValueError('Incomplete score table')
    for name in PROFILES:
        if table['original_or_translation'][name]['tp']<table['original'][name]['tp']:
            raise ValueError('OR unexpectedly lost old TPs')
    if sha_file(ext)!=V65_SHA or sha_file(root/'model.joblib')!=V64_SHA:
        raise ValueError('Model changed during test')
    result=dict(experiment='MailGuard v67 translation A/B, RETROSPECTIVE ONLY',
                baseCohort='v66 previously first-use scored Turkish stemmed public mail',
                sampleSelection='384 smallest sha256(v67-label-blind-v1:identity), no class or outcome filter',
                selectedIdentityFingerprint=selected_hash,originalCohortSha256=manifest['fingerprintSha256'],
                modelSha256=dict(v64=V64_SHA,v62=V62_SHA,v65=V65_SHA),
                translator=dict(repo=TRANSLATOR,revision=TRANSLATOR_REVISION,
                                license='Apache-2.0',inputTokenLimit=192,decodedTokenLimit=160,
                                truncatedInputs=truncated),
                selected=len(chosen),spam=sum(r['label'] for r in chosen),
                ham=sum(1-r['label'] for r in chosen),
                results=table,pairedChanges={k:dict(v) for k,v in improvements.items()},
                fitOrCalibrateOnV66=False,changedFrozenModelOrThreshold=False,
                originalV66FirstUseResultsUnaffected=True,
                translatedTextOrMessageIdsSaved=False,
                caveat='Engineering replay of previously scored v66 TEST, not independent. Turkish corpus is stemmed/cleaned and does not represent MIME mail; translator may distort malicious indicators. 384 selected public emails cannot establish low FPR. Do NOT tune cutoffs from these outcomes or claim production performance.')
    (out/'result.json').write_text(json.dumps(result,indent=2))
    md=['# MailGuard v67 — Turkish to English translation (retrospective A/B)','',
        '**Previously evaluated v66 samples: exploratory replay, not new validation.**',
        'No new classifier fitting or threshold selection; exact SHA-pinned v64 and v65 checkpoints.',
        '',f'Selected {len(chosen)} label-blind Turkish messages ({result["spam"]} SPAM, {result["ham"]} HAM).',
        f'Translation token truncation: {truncated}; model {TRANSLATOR} at {TRANSLATOR_REVISION}.','',
        '| Model | Original Recall | Translated-only Recall | Original OR translated Recall | FP original / translated / OR |',
        '|---|---:|---:|---:|---:|']
    for p in PROFILES:
        m=[table[z][p] for z in SCENARIOS]
        md.append(f"| {p} | {m[0]['recall']:.2%} | {m[1]['recall']:.2%} | {m[2]['recall']:.2%} | {m[0]['fp']} / {m[1]['fp']} / {m[2]['fp']} (of {m[0]['ham']} HAM) |")
    md.extend(['','Paired changes (numbers are for investigation only):',''])
    for name,values in result['pairedChanges'].items():
        md.append(f"- {name}: {values}")
    md.extend(['','**Do not select an OR policy or tune cutoffs using this v66 TEST replay.**',
               'Further deployment requires development-only controls and a genuinely new independent bilingual SPAM/HAM test.',
               'No customer mail was used or transmitted. Only aggregated metrics and model references are saved.',''])
    (out/'result.md').write_text('\n'.join(md))
    print('\n'.join(md),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',default='frozen/manifest.json')
    p.add_argument('--prior',default='prior/blocked.sqlite')
    p.add_argument('--old-input',default='previous')
    p.add_argument('--model64',default='models/v64')
    p.add_argument('--model65',default='models/v65')
    p.add_argument('--output',default='report-v67')
    run(p.parse_args())
