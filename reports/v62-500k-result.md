# MailGuard v62: 500000 messages

Previously inspected historical public corpus, predominantly spam. The full replay includes development messages and is not an independent quality estimate. Use the no-development-overlap subset for the engineering comparison; it is also previously inspected and is not customer validation. Distinct identity/body/template keys do not guarantee semantic independence. Few legitimate messages limit FP evidence.

| Scope | Profile | Count | Spam recall | FP / HAM | HAM FPR | 95% FPR upper bound |
|---|---|---:|---:|---:|---:|---:|
| all | conservative | 500000 | 71.8472% | 0 / 775 | 0.0000% | 0.4932% |
| all | high-recall | 500000 | 91.2388% | 0 / 775 | 0.0000% | 0.4932% |
| no-development-overlap | conservative | 351485 | 71.2822% | 0 / 521 | 0.0000% | 0.7319% |
| no-development-overlap | high-recall | 351485 | 90.6070% | 0 / 521 | 0.0000% | 0.7319% |
| development-overlap | conservative | 148515 | 73.1845% | 0 / 254 | 0.0000% | 1.4899% |
| development-overlap | high-recall | 148515 | 92.7344% | 0 / 254 | 0.0000% | 1.4899% |
