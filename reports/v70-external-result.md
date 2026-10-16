# MailGuard v70 — English external first-use CSDMC2010 transfer

**Frozen v69 model; v70 policies selected exclusively on prior English CAL.**
No v70 model training or optimization on external challenge labels.

Eligible external email: 4049 (1305 SPAM / 2744 HAM).
Known-archive overlaps excluded before scoring: {}.

| Policy | Recall | TP / SPAM | FP / HAM | FPR |
|---|---:|---:|---:|---:|
| v69 / zero-cal-fp | 29.12% | 380 / 1305 | 1 / 2744 | 0.036% |
| v69 / research-0p1pct-cal-fpr | 55.17% | 720 / 1305 | 9 / 2744 | 0.328% |
| v70 / conservative-0cal-fp | 35.48% | 463 / 1305 | 3 / 2744 | 0.109% |
| v70 / sensitivity-0p05pct-cal-fp | 56.78% | 741 / 1305 | 14 / 2744 | 0.510% |

**Do not tune on these external TEST labels, do not infer Russian production quality.**
2010 repackaged public messages are not representative of today's customer stream.
Zero observed false positives on this small HAM pool cannot establish 0.01% FPR.
The owner repo has MIT license but original corpus rights need separate verification.
Raw mail text and identifiers are not saved in GitHub artifacts.
