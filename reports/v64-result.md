# MailGuard v64: frozen 500k and large HAM evaluation

All TEST sources have already been inspected: this is a repeated engineering comparison, not independent customer validation. The full 500k replay includes development messages; use its no-development-overlap subset for quality. The 75271 HAM set excludes the full TRAIN/SELECT/CAL partitions used by both versions. Original labels are preserved. No TEST labels fit, choose or calibrate v64; thresholds remain fixed.

| Dataset | Scope | Profile | Count | Recall | FP / HAM |
|---|---|---|---:|---:|---:|
| 500k | all | conservative | 500000 | 87.0143% | 0 / 775 |
| 500k | all | high-recall | 500000 | 95.2715% | 1 / 775 |
| 500k | no-development-overlap | conservative | 351485 | 86.6733% | 0 / 521 |
| 500k | no-development-overlap | high-recall | 351485 | 94.9280% | 1 / 521 |
| ham | all | conservative | 75271 | n/a | 7 / 75271 |
| ham | all | high-recall | 75271 | n/a | 76 / 75271 |
| ham | additional | conservative | 47258 | n/a | 7 / 47258 |
| ham | additional | high-recall | 47258 | n/a | 61 / 47258 |
| may | all | conservative | 4146 | 64.4665% | 1 / 313 |
| may | all | high-recall | 4146 | 92.5385% | 1 / 313 |
| summer | all | conservative | 11831 | 55.2976% | 0 / 1968 |
| summer | all | high-recall | 11831 | 83.8386% | 0 / 1968 |
| october | all | conservative | 122 | 61.4754% | 0 / 0 |
| october | all | high-recall | 122 | 81.9672% | 0 / 0 |

## v62 / v64 on identical main inputs

| Version | Profile | 500k holdout recall | FP / 75271 HAM |
|---|---|---:|---:|
| v62 | conservative | 71.2822% | 0 |
| v64 | conservative | 86.6733% | 7 |
| v62 | high-recall | 90.6070% | 164 |
| v64 | high-recall | 94.9280% | 76 |
