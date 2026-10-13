# MailGuard v60 — Unicode neural ensemble

Model and thresholds frozen before new May TEST. Prior v59 TEST never enters fitting.

| Dataset | Model/profile | Mail | HAM | Spam recall | FP | FN |
|---|---|---:|---:|---:|---:|---:|
| New TEST | v60 conservative | 4146 | 313 | 7.10% | 0 | 3561 |
| New TEST | v59 conservative | 4146 | 313 | 58.02% | 0 | 1609 |
| New TEST | v60 high-recall | 4146 | 313 | 78.53% | 5 | 823 |
| New TEST | v59 high-recall | 4146 | 313 | 71.61% | 5 | 1088 |
| Reused v59 TEST | v60 conservative | 11831 | 1968 | 9.08% | 0 | 8967 |
| Reused v59 TEST | v59 conservative | 11831 | 1968 | 47.39% | 0 | 5189 |
| Reused v59 TEST | v60 high-recall | 11831 | 1968 | 71.58% | 0 | 2803 |
| Reused v59 TEST | v59 high-recall | 11831 | 1968 | 66.13% | 1 | 3341 |
| Reused English/CSDMC | v60 conservative | 44615 | 24017 | 14.36% | 2 | 17640 |
| Reused English/CSDMC | v59 conservative | 44615 | 24017 | 67.61% | 15 | 6672 |
| Reused English/CSDMC | v60 high-recall | 44615 | 24017 | 89.72% | 42 | 2117 |
| Reused English/CSDMC | v59 high-recall | 44615 | 24017 | 92.01% | 58 | 1645 |

New May spam-trap families and unused original SpamAssassin messages, after prior-source blocking. HAM count may be small; separate reused Chinese and English regression results must not be presented as new. Historical public mail is not customer mail; semantic relatives may remain.

Publication: experimental only. High-recall improves new spam, but legacy recall falls and conservative collapses. Do not replace v59/v57 globally.
