# MailGuard v65 — verified provenance and FPR audit

**This is an audit, NOT a new benchmark or trained model.**

v65 source-disjoint adaptation is not the same as independent first-use evidence.
All checked v65 sources already appeared in v64 first-use experiments.

| Source | Status | Messages | Spam | Recall | FP/HAM |
|---|---|---:|---:|---:|---:|
| GenSpam | trained_or_calibrated_by_v65 | 3719 | 2100 | 36.90% | 0/1619 |
| Nazario-2025 | previously_scored_source | 403 | 403 | 21.34% | 0/0 |
| Published-ASCII-mail | trained_or_calibrated_by_v65 | 1304 | 804 | 66.17% | 2/500 |
| SpamArchive-2013 | previously_scored_source | 108922 | 108922 | 74.74% | 0/0 |
| SpamArchive-2014 | previously_scored_source | 95705 | 95705 | 73.49% | 0/0 |
| SpamArchive-2015 | previously_scored_source | 109426 | 109426 | 80.73% | 0/0 |
| SpamArchive-2016 | trained_or_calibrated_by_v65 | 132623 | 132623 | 89.82% | 0/0 |

## False positives
v65: 2/2119 HAM; FPR 0.0944%; 95% Wilson upper bound 0.3435%.
Target FPR: 0.0100%. With zero FP, at least 29,956 independent HAM are needed for a one-sided 95% binomial bound.
Two 100000-email archive subsets are all SPAM; their FP is undefined, not zero-risk.

## Acceptance
- Previously unused sources scored for v65: **0**.
- Demonstrated target FPR: **False**.
- Generalization to arbitrary new 100000-email mixes: **not demonstrated**.

Next: collect an independent original-source corpus with its native HAM/SPAM labels; freeze provenance and exclusions before scoring. Maintain separate first-use report and do not re-label already-inspected data as independent.
