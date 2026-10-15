# MailGuard v68 — Russian/English only (research baseline)

This is a retrospective engineering benchmark. RU is Telegram chat, NOT emails; EN mail corpus includes sources already examined in earlier MailGuard versions. The 10000-item sample is grouped within two providers, NOT arbitrary independent suppliers. Neither production 90% recall nor 0.01% FPR is validated. Do not use its TEST to tune thresholds.

Model SHA256: 92a38fef77102761b172d973a63720bed1f52a7d9e8ea8e97424c63baabc4f3d

| Language | Profile | N | SPAM | Recall | FP / HAM | Wilson 95% FPR upper |
|---|---|---:|---:|---:|---:|---:|
| ru | conservative | 5000 | 1690 | 71.72% | 0 / 3310 | 0.116% |
| ru | research-high-recall | 5000 | 1690 | 96.15% | 7 / 3310 | 0.436% |
| en | conservative | 5000 | 2318 | 11.09% | 0 / 2682 | 0.143% |
| en | research-high-recall | 5000 | 2318 | 67.77% | 2 / 2682 | 0.272% |
| ALL | conservative | 10000 | 4008 | 36.65% | 0 / 5992 | 0.064% |
| ALL | research-high-recall | 10000 | 4008 | 79.74% | 9 / 5992 | 0.285% |

This is 10000 **grouped samples within known source families** (5000 each language), not a random independent stream or an arbitrary source acceptance test.
Russian samples are Telegram chat, not emails; English samples include historically trained source pools.
90% target is NOT considered achieved unless independently tested on fresh source-disjoint 10k mail cohorts and with adequate independent HAM.
All model and score artifacts must be treated as research-only.
