"""Compare hash-only C# runtime receipts. This never establishes deployment acceptance."""
import argparse
import hashlib
import json
import math
from pathlib import Path

DECISIONS = {'Normal', 'Spam', 'Unsure'}


def _hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def load_receipt(path):
    with Path(path).open(encoding='utf-8') as stream:
        lines = [json.loads(line) for line in stream if line.strip()]
    if not lines or lines[0].get('kind') != 'metadata' or lines[0].get('schemaVersion') != 1:
        raise ValueError('Expected runtime receipt metadata version 1')
    metadata = lines[0]
    if metadata.get('rspamdContext') != 'absent' or metadata.get('rawContentSaved') is not False or type(metadata.get('classificationAvailable')) is not bool:
        raise ValueError('Unsupported runtime receipt provenance')
    records = {}
    for row in lines[1:]:
        identity = row.get('identity')
        if row.get('kind') != 'message' or not _hash(identity) or identity in records:
            raise ValueError('Invalid or duplicate message identity')
        if row.get('label') not in (None, 'spam', 'ham') or type(row.get('parseSuccess')) is not bool:
            raise ValueError('Invalid message label or parser status')
        if row['parseSuccess']:
            parsed = row.get('parsed')
            if not isinstance(parsed, dict) or any(not _hash(parsed.get(k)) for k in
                ('textSha256', 'subjectSha256', 'fromSha256', 'toSha256')) or not _hash(row.get('featureSha256')):
                raise ValueError('Invalid parser or feature hash')
            if metadata.get('classificationAvailable'):
                probability = row.get('probability')
                if row.get('decision') not in DECISIONS or isinstance(probability, bool) or not isinstance(probability, (int, float)) or not math.isfinite(probability) or not 0 <= probability <= 1:
                    raise ValueError('Invalid classification result')
        records[identity] = row
    if metadata.get('selectedMessageCount') != len(records):
        raise ValueError('Receipt count does not match its records')
    if not records:
        raise ValueError('Empty runtime receipt')
    return metadata, records


def compare(baseline_metadata, baseline, candidate_metadata, candidate):
    if set(baseline) != set(candidate):
        raise ValueError('Baseline and candidate message identities differ')
    if baseline_metadata.get('featureDimension') != candidate_metadata.get('featureDimension'):
        raise ValueError('Feature dimensions differ')
    if baseline_metadata.get('classificationAvailable') != candidate_metadata.get('classificationAvailable'):
        raise ValueError('Baseline and candidate inference availability differs')
    for identity in baseline:
        if baseline[identity].get('label') != candidate[identity].get('label'):
            raise ValueError('Baseline and candidate labels differ')
    model_available = bool(baseline_metadata.get('classificationAvailable') and candidate_metadata.get('classificationAvailable'))
    if model_available and (not _hash(baseline_metadata.get('modelSha256')) or baseline_metadata['modelSha256'] != candidate_metadata.get('modelSha256')):
        raise ValueError('Classification comparison requires the exact same model bytes')
    result = dict(messages=len(baseline), baseline_parse_errors=0, candidate_parse_errors=0,
                  parser_status_changes=0, parsed_output_changes=0, feature_changes=0,
                  classification_comparison_available=model_available, prediction_comparable_messages=0,
                  baseline_spam_detections_lost=0, spam_detections_gained=0,
                  false_positives_removed=0, false_positives_added=0, decision_changes=0,
                  baseline_spam_detections_unverifiable_due_candidate_error=0,
                  unlabelled_messages=0, raw_content_saved=False,
                  deployment_acceptance_established=False,
                  deployment_acceptance_not_established_reasons=['Frozen trusted Rspamd context is absent',
                      'Receipt provenance does not establish approved deployed model or independent corpus'])
    transitions = {}
    for identity in sorted(baseline):
        old, new = baseline[identity], candidate[identity]
        a, b = old['parseSuccess'], new['parseSuccess']
        result['baseline_parse_errors'] += not a
        result['candidate_parse_errors'] += not b
        result['parser_status_changes'] += a != b
        if old.get('label') is None:
            result['unlabelled_messages'] += 1
        if not (a and b):
            if model_available and a and not b and old.get('label') == 'spam' and old.get('decision') == 'Spam':
                result['baseline_spam_detections_unverifiable_due_candidate_error'] += 1
            continue
        result['parsed_output_changes'] += old['parsed'] != new['parsed']
        result['feature_changes'] += old['featureSha256'] != new['featureSha256']
        if not model_available:
            continue
        result['prediction_comparable_messages'] += 1
        old_decision, new_decision = old['decision'], new['decision']
        transitions[old_decision + '->' + new_decision] = transitions.get(old_decision + '->' + new_decision, 0) + 1
        result['decision_changes'] += old_decision != new_decision
        was_spam, is_spam = old_decision == 'Spam', new_decision == 'Spam'
        if old['label'] == 'spam':
            result['baseline_spam_detections_lost'] += was_spam and not is_spam
            result['spam_detections_gained'] += not was_spam and is_spam
        elif old['label'] == 'ham':
            result['false_positives_removed'] += was_spam and not is_spam
            result['false_positives_added'] += not was_spam and is_spam
    result['decision_transitions'] = transitions
    result['baseline_core_assembly_sha256'] = baseline_metadata.get('coreAssemblySha256')
    result['candidate_core_assembly_sha256'] = candidate_metadata.get('coreAssemblySha256')
    result['classification_comparison_complete'] = bool(model_available and
        result['prediction_comparable_messages'] == result['messages'])
    if not model_available:
        result['deployment_acceptance_not_established_reasons'].append('Matching .mg model inference is absent')
        # Absence of inference is unknown, not evidence of zero lost detections.
        for key in ('baseline_spam_detections_lost', 'spam_detections_gained',
                    'false_positives_removed', 'false_positives_added', 'decision_changes',
                    'baseline_spam_detections_unverifiable_due_candidate_error'):
            result[key] = None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    old_metadata, old = load_receipt(args.baseline)
    new_metadata, new = load_receipt(args.candidate)
    result = compare(old_metadata, old, new_metadata, new)
    result['baseline_receipt_sha256'] = hashlib.sha256(args.baseline.read_bytes()).hexdigest()
    result['candidate_receipt_sha256'] = hashlib.sha256(args.candidate.read_bytes()).hexdigest()
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
