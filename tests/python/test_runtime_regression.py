import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from compare_runtime_regression import compare, load_receipt


def metadata(classify=False):
    return dict(kind='metadata', schemaVersion=1, featureDimension=131072, selectedMessageCount=2,
                modelSha256='f' * 64 if classify else None, classificationAvailable=classify,
                rspamdContext='absent', rawContentSaved=False)


def row(identity, label, decision='Spam'):
    return dict(kind='message', identity=identity * 64, label=label, parseSuccess=True,
                parsed=dict(textSha256='a' * 64, subjectSha256='b' * 64, fromSha256='c' * 64, toSha256='d' * 64,
                            urlCount=0, attachmentCount=0, hasHtml=False, rawLength=100),
                featureSha256='e' * 64, featureCount=4, decision=decision, probability=.9)


class RuntimeRegressionTests(unittest.TestCase):
    def setUp(self):
        self.rows = {'1' * 64: row('1', 'spam'), '2' * 64: row('2', 'ham')}

    def test_parser_changes_without_a_model_never_claim_deployment(self):
        candidate = copy.deepcopy(self.rows)
        candidate['1' * 64]['parsed']['textSha256'] = 'f' * 64
        candidate['1' * 64]['featureSha256'] = 'f' * 64
        result = compare(metadata(), self.rows, metadata(), candidate)
        self.assertEqual(result['parsed_output_changes'], 1)
        self.assertEqual(result['feature_changes'], 1)
        self.assertFalse(result['classification_comparison_available'])
        self.assertIsNone(result['baseline_spam_detections_lost'])
        self.assertFalse(result['deployment_acceptance_established'])

    def test_lost_tp_and_removed_fp_are_individual_transitions(self):
        candidate = copy.deepcopy(self.rows)
        candidate['1' * 64]['decision'] = 'Unsure'
        candidate['2' * 64]['decision'] = 'Normal'
        result = compare(metadata(True), self.rows, metadata(True), candidate)
        self.assertEqual(result['baseline_spam_detections_lost'], 1)
        self.assertEqual(result['false_positives_removed'], 1)
        self.assertEqual(result['decision_changes'], 2)
        self.assertFalse(result['deployment_acceptance_established'])

    def test_missing_identity_label_and_model_drift_are_rejected(self):
        for change in ('missing', 'label', 'model'):
            candidate = copy.deepcopy(self.rows); meta = metadata(True)
            if change == 'missing': candidate.pop('1' * 64)
            if change == 'label': candidate['1' * 64]['label'] = 'ham'
            if change == 'model': meta['modelSha256'] = 'a' * 64
            with self.assertRaises(ValueError):
                compare(metadata(True), self.rows, meta, candidate)

    def test_duplicate_identity_and_invalid_probability_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'receipt.jsonl'
            for records in ([self.rows['1' * 64]] * 2,
                            [dict(self.rows['1' * 64], probability=float('nan')), self.rows['2' * 64]]):
                path.write_text('\n'.join(json.dumps(x) for x in [metadata(True), *records]))
                with self.assertRaises(ValueError): load_receipt(path)

    def test_parser_failures_are_explicit_not_treated_as_ham(self):
        candidate = copy.deepcopy(self.rows)
        candidate['1' * 64] = dict(kind='message', identity='1' * 64, label='spam', parseSuccess=False, errorType='InvalidDataException')
        result = compare(metadata(True), self.rows, metadata(True), candidate)
        self.assertEqual(result['candidate_parse_errors'], 1)
        self.assertEqual(result['prediction_comparable_messages'], 1)
        self.assertEqual(result['baseline_spam_detections_lost'], 0)
        self.assertEqual(result['baseline_spam_detections_unverifiable_due_candidate_error'], 1)
        self.assertFalse(result['classification_comparison_complete'])
        self.assertFalse(result['deployment_acceptance_established'])


if __name__ == '__main__':
    unittest.main()
