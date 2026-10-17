# MailGuard v74: trained multilingual Transformer (not a sparse classifier)

RETROSPECTIVE: reuses v68/v69 fixed 10k TEST; no independent first-use claim.

Pretrained encoder: microsoft/Multilingual-MiniLM-L12-H384; tokenizer: FacebookAI/xlm-roberta-base.
Three of twelve Transformer layers + classification head really updated.
Trainable parameters: 5472002; optimizer steps: 438.
RU samples = Telegram CHAT, not real email; EN sources already used.
Checkpoint SHA256: 95363312973c9a6e1544248fd87665c4c860fbad409aa579a86c11397b35fa78

| Language | Policy | Recall | TP/SPAM | FP/HAM | FPR |
|---|---|---:|---:|---:|---:|
| ru | zero-cal-fp | 88.28% | 1492/1690 | 1/3310 | 0.030% |
| ru | research-0p1pct-cal-fpr | 89.82% | 1518/1690 | 1/3310 | 0.030% |
| en | zero-cal-fp | 27.18% | 630/2318 | 2/2682 | 0.075% |
| en | research-0p1pct-cal-fpr | 29.77% | 690/2318 | 11/2682 | 0.410% |
| ALL | zero-cal-fp | 52.94% | 2122/4008 | 3/5992 | 0.050% |
| ALL | research-0p1pct-cal-fpr | 55.09% | 2208/4008 | 12/5992 | 0.200% |

Frozen v69 reference on exactly the same historical TEST:
- zero-cal-fp: 77.62% recall, 0/5992 FP
- research-0p1pct-cal-fpr: 92.07% recall, 10/5992 FP

Do NOT promote to customer based on reused test. Need genuinely unseen Russian MIME mail, diverse unseen English mail and at least ~30k independent HAM with zero FP for 0.01% FPR confidence.
