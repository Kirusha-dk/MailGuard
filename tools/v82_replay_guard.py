"""Classify reproducibility failures without relaxing acceptance criteria."""
import math
from audit_v82_errors import paired


def locked_decision_replay(rows, reference, observed, threshold):
    if len(rows) != len(reference) or len(reference) != len(observed):
        raise ValueError('Reference score alignment drift')
    if any(not math.isfinite(float(x)) or not 0 <= x <= 1
           for values in (reference, observed) for x in values):
        raise ValueError('Invalid probability in replay')
    result = paired(rows, reference, observed, threshold, threshold)
    keys = ('baseline_spam_detections_lost', 'spam_detections_gained',
            'false_positives_removed', 'false_positives_added')
    result['all_individual_decisions_reproduced'] = all(result[k] == 0 for k in keys)
    result['max_absolute_score_difference'] = max(abs(a - b) for a, b in zip(reference, observed))
    result['exactly_equal_scores'] = sum(a == b for a, b in zip(reference, observed))
    result['compares_two_v82_development_checkpoints'] = False
    result['compares_same_checkpoint_replay'] = True
    return result


def replay_differences(expected, observed):
    keys = ('n', 'spam', 'ham', 'tp', 'fp', 'threshold', 'log_loss')
    if any(k not in expected or k not in observed for k in keys):
        raise ValueError('Incomplete replay metrics')
    if any(not math.isfinite(float(report[key])) for report in (expected, observed) for key in keys):
        raise ValueError('Nonfinite replay metrics')
    counts = {k: dict(expected=expected[k], observed=observed[k])
              for k in keys[:5] if expected[k] != observed[k]}
    numeric = {k: dict(expected=expected[k], observed=observed[k], delta=observed[k] - expected[k])
               for k in keys[5:] if expected[k] != observed[k]}
    return dict(count_differences=counts, numerical_differences=numeric,
                counts_match=not counts, exact_metrics_match=not counts and not numeric,
                interpretation='decision_counts_changed' if counts else
                               'numerical_replay_difference_requires_investigation' if numeric else 'exact_metrics_match',
                training_authorized=False)


def require_exact_reference_scores(expected, observed, identities, observed_identities):
    if len(expected) != len(observed) or identities != observed_identities or len(identities) != len(expected):
        raise ValueError('Reference score alignment drift')
    if any(not math.isfinite(float(x)) for values in (expected, observed) for x in values):
        raise ValueError('Nonfinite reference scores')
    if any(a != b for a, b in zip(expected, observed)):
        raise ValueError('Numerical runtime drift: reference scores differ; stop before optimization')
