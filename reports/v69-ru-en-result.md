# MailGuard v69 — RU/EN NBSVM A/B research

**REUSE of v68 test IDs. NOT NEW INDEPENDENT 10K.**

REUSED V68 TEST: retrospective, NOT new or independent. RU is Telegram chat, NOT emails; EN contains familiar source families. No claims for arbitrary emails or 0.01% production FPR. Do not tune using these TEST results.

Model SHA256: 1a0b228cf86b43d06a62a98e4dc1039bb1788c4eeb14ea77efaa6eb1d9ddb228

| Language | Profile | Recall | TP / SPAM | FP / HAM | FPR |
|---|---|---:|---:|---:|---:|
| ru | zero-cal-fp | 94.85% | 1603 / 1690 | 0 / 3310 | 0.000% |
| ru | research-0p1pct-cal-fpr | 97.51% | 1648 / 1690 | 6 / 3310 | 0.181% |
| en | zero-cal-fp | 65.06% | 1508 / 2318 | 0 / 2682 | 0.000% |
| en | research-0p1pct-cal-fpr | 88.09% | 2042 / 2318 | 4 / 2682 | 0.149% |
| ALL | zero-cal-fp | 77.62% | 3111 / 4008 | 0 / 5992 | 0.000% |
| ALL | research-0p1pct-cal-fpr | 92.07% | 3690 / 4008 | 10 / 5992 | 0.167% |

Thresholds and expert choice were frozen using CAL only. TEST outcomes must not determine the next version's thresholds.
Russian Telegram is only a linguistic proxy for true mail.
At least ~30k independent HAM with 0 FP is needed for a ~0.01% one-sided 95% FPR bound; current 10k is too small.
