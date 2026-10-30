"""Explicit observed rates and one-sided exact binomial uncertainty."""
from scipy.stats import beta


def upper_rate(events, trials):
    if trials == 0:
        return None
    if not 0 <= events <= trials:
        raise ValueError('Invalid event/trial counts')
    return 1. if events == trials else float(beta.ppf(.95, events + 1, trials - events))


def enrich(metric):
    row = dict(metric)
    n, ham, spam, fp = (row[k] for k in ('n', 'ham', 'spam', 'fp'))
    row['recall'] = row['tp'] / spam if spam else None
    row['fpr'] = fp / ham if ham else None
    row['fp_per_million_ham_point_estimate'] = fp / ham * 1e6 if ham else None
    row['fp_per_million_all_point_estimate'] = fp / n * 1e6 if n else None
    upper_ham, upper_all = upper_rate(fp, ham), upper_rate(fp, n)
    row['fpr_upper_95_per_million_ham_iid_assumption'] = None if upper_ham is None else upper_ham * 1e6
    row['fp_upper_95_per_million_all_iid_assumption'] = None if upper_all is None else upper_all * 1e6
    row['joint_98_and_10ppm_point_targets_met'] = bool(ham and spam and row['recall'] >= .98 and fp / ham <= .00001)
    row['joint_98_and_10ppm_all_point_targets_met'] = bool(ham and spam and row['recall'] >= .98 and fp / n <= .00001)
    return row


def summary_extra(summary):
    metric = enrich(summary['metrics'])
    return {'metrics': metric, 'goal': {
        'target_spam_recall': .98, 'primary_fp_denominator': 'ham',
        'target_fp_per_million_ham': 10, 'target_fp_per_million_all_messages': 10,
        'observed_joint_goal_on_scored_mail': metric['joint_98_and_10ppm_point_targets_met'],
        'observed_joint_goal_per_all_messages': metric['joint_98_and_10ppm_all_point_targets_met'],
        'at_least_one_million_ham_scored': metric['ham'] >= 1_000_000,
        'upper_95_fp_ham_within_10ppm_iid_assumption': bool(metric['ham'] and upper_rate(metric['fp'], metric['ham']) <= .00001),
        'at_least_one_million_messages_scored': metric['n'] >= 1_000_000,
        'upper_95_fp_all_within_10ppm_iid_assumption': bool(metric['ham'] and upper_rate(metric['fp'], metric['n']) <= .00001),
        'arbitrary_future_mail_guaranteed': False,
        'note': 'Point estimates are not a million-message test. Binomial bounds assume independent representative messages.'}}
