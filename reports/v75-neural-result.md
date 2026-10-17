# MailGuard v75 neural continuation + v69 safeguard

Retrospective reused fixed 10k, NOT an independent new-source test.
Train 10000 messages from verified v74 checkpoint;
context 160 tokens (v74 had 96), batch 16, one additional epoch.
Checkpoint SHA256: f4c4dfa7a25b7b4daa60af0807a5f404b8969f594cab1b7e59b2842e73771698.
The neural mixing weight is selected on SELECT; the threshold on family-disjoint CAL.

| Language | Profile | Model | Recall | TP/SPAM | FP/HAM | FPR |
|---|---|---|---:|---:|---:|---:|
| ru | zero-cal-fp | frozen v69 | 94.85% | 1603/1690 | 0/3310 | 0.000% |
| ru | zero-cal-fp | v75 gated NN | 95.98% | 1622/1690 | 0/3310 | 0.000% |
| ru | research-0p1pct-cal-fpr | frozen v69 | 97.51% | 1648/1690 | 6/3310 | 0.181% |
| ru | research-0p1pct-cal-fpr | v75 gated NN | 95.98% | 1622/1690 | 0/3310 | 0.000% |
| en | zero-cal-fp | frozen v69 | 65.06% | 1508/2318 | 0/2682 | 0.000% |
| en | zero-cal-fp | v75 gated NN | 87.92% | 2038/2318 | 4/2682 | 0.149% |
| en | research-0p1pct-cal-fpr | frozen v69 | 88.09% | 2042/2318 | 4/2682 | 0.149% |
| en | research-0p1pct-cal-fpr | v75 gated NN | 89.30% | 2070/2318 | 4/2682 | 0.149% |
| ALL | zero-cal-fp | frozen v69 | 77.62% | 3111/4008 | 0/5992 | 0.000% |
| ALL | zero-cal-fp | v75 gated NN | 91.32% | 3660/4008 | 4/5992 | 0.067% |
| ALL | research-0p1pct-cal-fpr | frozen v69 | 92.07% | 3690/4008 | 10/5992 | 0.167% |
| ALL | research-0p1pct-cal-fpr | v75 gated NN | 92.12% | 3692/4008 | 4/5992 | 0.067% |

No promise of 90% on arbitrary unseen mail, or 0.01% FPR.
All Russian examples are Telegram chats. Test already used repeatedly.
Do not promote versions by choosing the winner on the old TEST.
