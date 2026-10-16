# MailGuard v73 — previously unscored 10k republisher sample

**Tested without retraining or adjusting frozen v69/v72 cutoffs.**
Not a new independent *original provider*: compiled corpora can have overlap beyond hashed-content screening.

Publisher: locuoco/the-biggest-spam-ham-phish-email-dataset-300000 @ 22fa6d1c1362518821bfb25564654348014303e4
Available messages scanned: 365448.
Prior exact exclusions: 241; prefix exclusions: 267.
Fixed sample: 10000, SPAM/PHISH 5364, HAM 4636.

| Model | Policy | Recall | TP / SPAM | FP / HAM | FPR |
|---|---|---:|---:|---:|---:|
| v69 | zero-cal-fp | 57.53% | 3086 / 5364 | 3 / 4636 | 0.065% |
| v69 | research-0p1pct-cal-fpr | 79.98% | 4290 / 5364 | 28 / 4636 | 0.604% |
| v72 | zero-cal-fp | 50.69% | 2719 / 5364 | 2 / 4636 | 0.043% |
| v72 | research-0p1pct-cal-fpr | 68.53% | 3676 / 5364 | 8 / 4636 | 0.173% |

This does not demonstrate 90% on arbitrary incoming 10k mail, and not a production 0.01% FPR guarantee. Missing verified Russian MIME samples and large independent HAM data.
Reprocessed/aggregated mail may share original underlying corpora with training. Do not retune models on the outcomes of this challenge.
No raw email text, predictions or message IDs saved.
