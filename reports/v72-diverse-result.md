# MailGuard v72 — diverse-source English adaptation, retrospective v71 test

**Existing v71 TEST used AGAIN; this is NOT another first-use result.**

Locked v72 SHA256: cc99c9e2b2c1d6e36fdfa0c8404c74191e7cfb3a4f2316cad9722b06146eb737
External CSDMC train/cal: 2840 / 1209 (family-disjoint).
External train repeated 4 times alongside 45000 in-domain English training samples.
Both CAL source FP caps applied jointly. Old RU v69 model unchanged.

| Source | Policy | Recall | TP / SPAM | FP / HAM | FPR |
|---|---|---:|---:|---:|---:|
| v69 | zero-cal-fp | 62.95% | 2995 / 4758 | 2 / 5242 | 0.038% |
| v69 | research-0p1pct-cal-fpr | 84.62% | 4026 / 4758 | 11 / 5242 | 0.210% |
| v72 | zero-cal-fp | 60.45% | 2876 / 4758 | 1 / 5242 | 0.019% |
| v72 | research-0p1pct-cal-fpr | 81.21% | 3864 / 4758 | 6 / 5242 | 0.114% |

This is not production validation; never change the model to maximize the historical v71 TEST. Acquire different genuinely first-use source before promotion.
CSDMC original data license uncertain. No client email used, no raw email stored in artifacts.
