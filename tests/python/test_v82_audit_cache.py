import json
import gzip
import importlib.util
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from audit_v82_errors import describe, paired, seen_train
from summarize_v82_audit_cache import analyze, fingerprint, log_loss, preservation_frontier, validate_alignment


class CachedAuditTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(identity=str(i), source='a' if i < 2 else 'b', y=i % 2,
                          family=str(i), text='PRIVATE_BODY ' + 'word ' * (i * 50)) for i in range(6)]

    def test_scalar_frontier_preserves_individual_tp_and_includes_ties(self):
        old = [.8, .9, .1, .8, .1, .1]
        new = [.4, .2, .1, .7, .2, .9]
        result = preservation_frontier(self.rows, old, new, .5)
        self.assertEqual(result['threshold'], .2)
        paired = result['paired_select']
        self.assertEqual(paired['baseline_spam_detections_lost'], 0)
        self.assertEqual(paired['spam_detections_gained'], 1)
        self.assertEqual(paired['current']['fp'], 2)
        # Any higher cutoff loses the original TP with score .2.
        self.assertTrue(result['selected_on_same_development'])
        self.assertFalse(result['deployable_policy'])

    def test_permutation_labels_duplicates_and_invalid_probabilities_are_rejected(self):
        ids = [r['identity'] for r in self.rows]; labels = [r['y'] for r in self.rows]
        validate_alignment(self.rows, ids, labels, [.5] * 6)
        for identities, ys, scores in ((ids[::-1], labels, [.5] * 6),
                                       (ids, labels[::-1], [.5] * 6),
                                       (ids, labels, [1.01] * 6),
                                       (ids, labels, [float('nan')] * 6)):
            with self.assertRaises(ValueError):
                validate_alignment(self.rows, identities, ys, scores)
        with self.assertRaises(ValueError):
            validate_alignment([self.rows[0]] * 2, [ids[0]] * 2, [0] * 2, [.5] * 2)

    def test_group_transitions_reconcile_and_no_message_body_is_saved(self):
        old = [.8, .9, .1, .8, .1, .1]; new = [.4, .2, .1, .7, .2, .9]
        result = analyze(self.rows, self.rows, old, new, new, .5, .5)
        for key in ('paired_by_source', 'paired_by_word_count'):
            self.assertEqual(sum(g['baseline_spam_detections_lost'] for g in result[key].values()), 1)
            self.assertEqual(sum(g['spam_detections_gained'] for g in result[key].values()), 1)
            self.assertEqual(sum(g['false_positives_removed'] for g in result[key].values()), 1)
        self.assertNotIn('PRIVATE_BODY', json.dumps(result))
        self.assertEqual(result['source_standardized_seen_train_fpr'], result['observed_select_fpr'])

    def test_loss_is_unweighted_and_handles_probability_endpoints(self):
        self.assertAlmostEqual(log_loss(self.rows, [.5] * 6), math.log(2))
        self.assertTrue(math.isfinite(log_loss(self.rows, [r['y'] for r in self.rows])))
        with self.assertRaises(ValueError): log_loss([], [])
        with self.assertRaises(ValueError): preservation_frontier(self.rows, [0.] * 6, [.5] * 6, .5)

    @unittest.skipUnless(importlib.util.find_spec('numpy'), 'CLI fixture needs numpy')
    def test_cli_saves_explicit_diagnostic_but_still_fails_strict_loss_check(self):
        import numpy as np
        script = Path(__file__).resolve().parents[2] / 'tools/summarize_v82_audit_cache.py'
        old = [.8, .9, .1, .8, .1, .1]; new = [.4, .2, .1, .7, .2, .9]
        sample, population = seen_train(self.rows, dict(epoch=0, cursor=6), 82, 6)
        train_scores = [new[int(r['identity'])] for r in sample]
        def report(scores, step, checksum):
            dev = describe(self.rows, scores, .5)['at_locked_development_threshold']
            dev.update(threshold=.5, log_loss=log_loss(self.rows, scores))
            return dict(development=dev, global_optimizer_steps=step, checkpoint_sha256=checksum,
                        epoch=0, cursor=6)
        base = report(old, 500, 'a' * 64); current = report(new, 1000, 'b' * 64)
        # Small drift must remain visible and must not produce a successful job.
        base['development']['log_loss'] += 1e-7
        reference = dict(campaign='fixture', global_optimizer_steps=1000, baseline_step=500,
                         baseline_checkpoint_sha256='a' * 64, current_checkpoint_sha256='b' * 64,
                         checkpoint_files_unchanged=True, source_report_counts_reproduced=True,
                         optimizer_steps_executed=0, cal_or_test_opened=False,
                         source_report=current['development'],
                         corpus_fingerprints=dict(train=fingerprint(self.rows), select=fingerprint(self.rows)),
                         seen_train_sample_fingerprint=fingerprint(sample), train_seen_population=population,
                         seen_train_sample=describe(sample, train_scores, .5), select=describe(self.rows, new, .5),
                         paired_select=paired(self.rows, old, new, .5, .5))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def write(path, value):
                target = root / path; target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(value))
            for cache, manifest, rows in (('.cache/v79/frozen', 'reports/v79-frozen.json', self.rows),
                                          ('.cache/v80/extra', 'reports/v80-extra-frozen.json', [])):
                write(Path(cache) / 'frozen.json', {}); write(manifest, {})
                for part in ('train', 'select'):
                    with gzip.open(root / cache / (part + '.jsonl.gz'), 'wt') as stream:
                        for row in rows: stream.write(json.dumps(row) + '\n')
            write('cache/audit.json', reference); write('reference.json', reference)
            write('base.json', base); write('current.json', current)
            write('request.json', dict(campaign='fixture', expected_min_step=1000, baseline_step=500,
                                      train_seen_limit=6, cached_audit_run_id=1,
                                      cached_reference_report='reference.json',
                                      cached_baseline_report='base.json', cached_current_report='current.json'))
            np.savez_compressed(root / 'cache/development-scores.npz',
                                select_identity=np.array([r['identity'] for r in self.rows]),
                                select_y=np.array([r['y'] for r in self.rows]), current=np.array(new),
                                baseline=np.array(old), train_identity=np.array([r['identity'] for r in sample]),
                                train_y=np.array([r['y'] for r in sample]), train_scores=np.array(train_scores))
            run = subprocess.run([sys.executable, str(script), '--request', 'request.json',
                                  '--cache', 'cache', '--out', 'result.json'], cwd=root,
                                 text=True, capture_output=True)
            self.assertNotEqual(run.returncode, 0)
            result = json.loads((root / 'result.json').read_text())
            self.assertFalse(result['source_report_log_losses_reproduced'])
            self.assertTrue(result['source_report_counts_reproduced'])
            self.assertEqual(result['status'], 'diagnostic_only_log_loss_replay_mismatch')
            self.assertEqual(result['optimizer_steps_executed'], 0)
            self.assertNotIn('PRIVATE_BODY', json.dumps(result))


if __name__ == '__main__':
    unittest.main()
