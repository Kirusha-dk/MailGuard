"""Development-only selection and an early stop that also protects epoch zero."""
import math


def should_stop(progress, patience=3):
    return progress['epoch'] >= 3 or progress['stalled'] >= patience


def choose_recovery(parent, candidate):
    """An unproven candidate cannot displace the retained parent.

    Thresholds are fitted on SELECT development only. Both fixed-threshold
    detection and FP at the same fitted recall must improve over the parent.
    """
    for metrics in (parent, candidate):
        fixed = metrics['fixed_parent_threshold']
        if (not metrics.get('is_development') or metrics.get('is_independent_test') or
                metrics['n'] != 22790 or metrics['spam'] != 10589 or metrics['ham'] != 12201 or
                fixed['threshold'] != 0.04897100105881691 or
                not all(math.isfinite(metrics[k]) for k in ('log_loss','recall','threshold'))):
            raise ValueError('Wrong development population or threshold')
    a = parent['fixed_parent_threshold']; b = candidate['fixed_parent_threshold']
    accepted = (b['tp'] >= a['tp'] and b['fp'] <= a['fp'] and candidate['fp'] <= parent['fp'] and
                candidate['log_loss'] <= parent['log_loss'] and
                (b['tp'] > a['tp'] or candidate['fp'] < parent['fp']))
    return dict(selected='repair' if accepted else 'retained-parent',
                repair_passed=accepted, selection_partition='SELECT development',
                external_benchmark_used_for_selection=False,
                reason='non-regression gate passed' if accepted else
                       'repair has not surpassed the retained parent; retain its exact adapter')
