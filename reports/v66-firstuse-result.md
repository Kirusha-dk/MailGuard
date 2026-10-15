# MailGuard v66 frozen first-use test

**No model fitting or threshold calibration on v66.**
The 1307 screened records were frozen in a different workflow job before downloading models.
Data-quality limitations: Not a gold production benchmark: publisher labels and Turkish preprocessing/unknown license limit interpretation. CW-L phishing-only. Semantic near-duplicates are not comprehensively excluded. No customer mail used. A trustworthy rare-FP claim requires >=30000 independent, validated HAM at zero FP.

| Group | Model | N | Spam | Recall | FP/HAM | 95% FPR upper |
|---|---|---:|---:|---:|---:|---:|
| all | v64-conservative | 1307 | 818 | 17.97% | 0 / 489 | 0.78% |
| all | v64-high-recall | 1307 | 818 | 35.45% | 0 / 489 | 0.78% |
| all | v65-conservative-rescue | 1307 | 818 | 30.81% | 0 / 489 | 0.78% |
| source/Turkish-stemmed | v64-conservative | 984 | 495 | 0.00% | 0 / 489 | 0.78% |
| source/Turkish-stemmed | v64-high-recall | 984 | 495 | 0.00% | 0 / 489 | 0.78% |
| source/Turkish-stemmed | v65-conservative-rescue | 984 | 495 | 0.00% | 0 / 489 | 0.78% |
| source/CW-L-phishing | v64-conservative | 323 | 323 | 45.51% | 0 / 0 | n/a |
| source/CW-L-phishing | v64-high-recall | 323 | 323 | 89.78% | 0 / 0 | n/a |
| source/CW-L-phishing | v65-conservative-rescue | 323 | 323 | 78.02% | 0 / 0 | n/a |
| script/latin | v64-conservative | 1307 | 818 | 17.97% | 0 / 489 | 0.78% |
| script/latin | v64-high-recall | 1307 | 818 | 35.45% | 0 / 489 | 0.78% |
| script/latin | v65-conservative-rescue | 1307 | 818 | 30.81% | 0 / 489 | 0.78% |

These 1307 messages are **not** a random sample of mail and do **not** prove FPR of 0.01%, any-100k generalization, or customer performance.
Even zero FP among 489 legitimate messages would give a wide FPR uncertainty interval.
Saved predictions only as aggregate metrics; no raw message text, identifiers or scores published.
