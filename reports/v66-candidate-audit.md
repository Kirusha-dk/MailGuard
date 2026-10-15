# MailGuard v66 candidate source inventory (no predictions)

Source revisions, download SHA256 and old TEST exclusion checks have been frozen.

| Source | Eligible HAM | Eligible SPAM |
|---|---:|---:|
| Turkish-stemmed | 489 | 495 |
| CW-L-phishing | 0 | 323 |

Within-source collisions: {'duplicate': 1406}. Prior overlap exclusions: {'priorDevelopOrEarlierBenchmarks': 20}.

**Not independent performance validation.** Not a gold production benchmark: publisher labels and Turkish preprocessing/unknown license limit interpretation. CW-L phishing-only. Semantic near-duplicates are not comprehensively excluded. No customer mail used. A trustworthy rare-FP claim requires >=30000 independent, validated HAM at zero FP.

The audit publishes counts and hashes only, no messages, train data or predictions.
