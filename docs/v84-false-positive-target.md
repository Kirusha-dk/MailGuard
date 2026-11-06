# False-positive investigation — 2026-11-05

User requested a reduction toward 100 false positives on the existing 12,201-HAM SELECT. This corresponds to FPR 0.8196%, not 10 per million. No such result has been demonstrated. SELECT is development and cannot establish independent transfer.

Verified daily aggregates:

| Model / operating threshold | TP / 10,589 SPAM | FP / 12,201 HAM | Recall | FPR |
| --- | ---: | ---: | ---: | ---: |
| v82 step 8,500, SELECT-fitted ~98% | 10,378 | 1,044 | 98.0074% | 8.5567% |
| v84 step 11,625, SELECT-fitted ~98% | 10,378 | 1,027 | 98.0074% | 8.4173% |
| v84 step 11,625, fixed parent threshold | 10,339 | 883 | 97.6391% | 7.2371% |

These aggregate TP totals do not establish preservation of individual detections. The first v84 session completed 50,000 unique rows and 3,125 new optimizer steps; this supersedes the earlier incomplete activation milestone.

In v82, Merged-public contains 8,646 / 12,201 HAM (70.86%) and 892 / 1,044 FP (85.44%). Its FPR is 10.32%, versus 4.28% on other sources. In v84's fitted-98% report it accounts for 831 FP / 8,646 HAM, versus 196 FP / 3,555 HAM elsewhere. The large source size explains part of its FP count, while its higher FPR is also real under the inherited labels. Incomplete provenance is a reason to investigate labels, not evidence that particular HAM labels are wrong. Never exempt this source or relabel errors to improve the metric.

The existing threshold is selected for ~98% recall, with no hard FP cap. Raising it can reduce FP at the cost of more missed spam. To quantify that tradeoff, `tools/audit_v84_fp_budget.py` performs read-only inference on the pinned SELECT and completed v84 checkpoint. It reproduces all published confusion counts, then computes thresholds for budgets 1,000 / 500 / 250 / 100 / 50 / 10 / 0 FP and reports actual TP/FP/FN/TN plus source breakdowns. Thresholds use SELECT HAM labels and are exploratory development fits. No gradients, request changes or production promotion are performed. Scores contain hashes, source labels and probabilities only; raw text is not exported.

The audit was started locally on the exact completed checkpoint SHA256 `71c7cfbe41d0c0b3b8369dc60d3970b6ce3b75d155f76d6240304dffae5a9425`. The environment restart left no running process or completed output on November6; the last confirmed progress was 6,176 / 22,790 rows and no scores had been persisted. This is an interrupted audit, not a completed result. Do not claim a verified 100-FP candidate until final JSON passes report-count and checkpoint-hash checks. The existing workflow now recovers the November5 audit once from that exact completed checkpoint, before November6 gradients, with a 30-minute ceiling; an audit failure does not prevent the independently validated training continuation. Score prefixes are saved atomically every256 rows, included in the state artifact, and checked against checkpoint and row identities on retry. No new cron or controller is used. No independent test was performed.

Completed daily v84 evaluations now also save hashes, sources, labels and probabilities in state artifacts, with a checkpoint-hash metadata sidecar. Exploratory FP-budget operating points are computed from those existing predictions without extra inference or gradients. This reporting addition changes neither optimizer/scheduler/RNG/cursor restoration, sampling, training loss, fixed parent threshold, fitted98 rule, nor early stopping. The frozen manifest is unchanged.

Run: https://github.com/Kirusha-dk/MailGuard/actions/runs/42025999444 . The original v82 and other language campaigns are unchanged. Current training already applies extra HAM cost and extra cost to observed TRAIN errors; that does not guarantee 100 FP. Further training experiments must prioritize verified difficult TRAIN HAM without placing SELECT errors into gradients, preserve state and frozen data, and measure recall as well as FPR. A change in sampling or loss requires an explicit experiment rather than pretending it is identical continuation.
