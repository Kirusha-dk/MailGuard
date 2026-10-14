# MailGuard v64: first-use public transfer test

Historical public mail unused in development and prior benchmarks before this experiment, not customer traffic. The ready 25971-message labelled-source preview is included again with the same unchanged models, so it is not another independent test. GenSpam is anonymized; published ASCII mail has removed headers; Nazario is a single manually labelled inbox. Bulk spam traps are weak spam labels and come from a previously used provider in different years. Exact fingerprints and campaign prefixes exclude known relatives, but semantic near-duplicates may remain. HAM sample size may not establish extremely rare production FP. No synthetic messages or duplicated refill are used.

Exactly 500000 new selected messages: 489263 SPAM, 10737 HAM. Both models and thresholds were fixed before collection. The same messages are evaluated by both versions.

| Cohort | Version | Profile | Count | Spam recall | FP / HAM | FPR | Wilson 95% FPR upper |
|---|---|---|---:|---:|---:|---:|---:|
| all | v62 | conservative | 500000 | 65.7751% | 0 / 10737 | 0.0000% | 0.0358% |
| all | v62 | high-recall | 500000 | 88.4111% | 88 / 10737 | 0.8196% | 1.0086% |
| all | v64 | conservative | 500000 | 73.9856% | 0 / 10737 | 0.0000% | 0.0358% |
| all | v64 | high-recall | 500000 | 85.8908% | 7 / 10737 | 0.0652% | 0.1345% |
| unseen-source | v62 | conservative | 19469 | 1.8469% | 0 / 8315 | 0.0000% | 0.0462% |
| unseen-source | v62 | high-recall | 19469 | 49.2290% | 85 / 8315 | 1.0222% | 1.2622% |
| unseen-source | v64 | conservative | 19469 | 5.6930% | 0 / 8315 | 0.0000% | 0.0462% |
| unseen-source | v64 | high-recall | 19469 | 30.4285% | 6 / 8315 | 0.0722% | 0.1574% |
| new-mail-known-domain | v62 | conservative | 6502 | 13.6765% | 0 / 2422 | 0.0000% | 0.1584% |
| new-mail-known-domain | v62 | high-recall | 6502 | 81.8382% | 3 / 2422 | 0.1239% | 0.3636% |
| new-mail-known-domain | v64 | conservative | 6502 | 24.6569% | 0 / 2422 | 0.0000% | 0.1584% |
| new-mail-known-domain | v64 | high-recall | 6502 | 62.2549% | 1 / 2422 | 0.0413% | 0.2335% |
| different-year-same-provider | v62 | conservative | 474029 | 67.7277% | 0 / 0 | n/a | n/a |
| different-year-same-provider | v62 | high-recall | 474029 | 89.3897% | 0 / 0 | n/a | n/a |
| different-year-same-provider | v64 | conservative | 474029 | 76.0171% | 0 / 0 | n/a | n/a |
| different-year-same-provider | v64 | high-recall | 474029 | 87.3993% | 0 / 0 | n/a | n/a |

## Source and script transfer

| Group | Version | Profile | Spam | Recall | FP / HAM |
|---|---|---|---:|---:|---:|
| script/cjk | v62 | conservative | 307775 | 99.4541% | 0 / 0 |
| script/cjk | v62 | high-recall | 307775 | 99.8798% | 0 / 0 |
| script/cjk | v64 | conservative | 307775 | 98.9310% | 0 / 0 |
| script/cjk | v64 | high-recall | 307775 | 99.7475% | 0 / 0 |
| script/latin | v62 | conservative | 166731 | 9.4260% | 0 / 10737 |
| script/latin | v62 | high-recall | 166731 | 70.4392% | 88 / 10737 |
| script/latin | v64 | conservative | 166731 | 33.5684% | 0 / 10737 |
| script/latin | v64 | high-recall | 166731 | 61.5788% | 7 / 10737 |
| script/other | v62 | conservative | 14757 | 0.0136% | 0 / 0 |
| script/other | v62 | high-recall | 14757 | 52.2735% | 0 / 0 |
| script/other | v64 | conservative | 14757 | 10.3680% | 0 / 0 |
| script/other | v64 | high-recall | 14757 | 71.5796% | 0 / 0 |
| source/GenSpam | v62 | conservative | 10751 | 1.8231% | 0 / 8315 |
| source/GenSpam | v62 | high-recall | 10751 | 49.6977% | 85 / 8315 |
| source/GenSpam | v64 | conservative | 10751 | 5.3390% | 0 / 8315 |
| source/GenSpam | v64 | high-recall | 10751 | 30.2577% | 6 / 8315 |
| source/Nazario-2025 | v62 | conservative | 403 | 2.4814% | 0 / 0 |
| source/Nazario-2025 | v62 | high-recall | 403 | 36.7246% | 0 / 0 |
| source/Nazario-2025 | v64 | conservative | 403 | 15.1365% | 0 / 0 |
| source/Nazario-2025 | v64 | high-recall | 403 | 34.9876% | 0 / 0 |
| source/Published-ASCII-mail | v62 | conservative | 4080 | 13.6765% | 0 / 2422 |
| source/Published-ASCII-mail | v62 | high-recall | 4080 | 81.8382% | 3 / 2422 |
| source/Published-ASCII-mail | v64 | conservative | 4080 | 24.6569% | 0 / 2422 |
| source/Published-ASCII-mail | v64 | high-recall | 4080 | 62.2549% | 1 / 2422 |
| source/SpamArchive-2013 | v62 | conservative | 108922 | 65.2577% | 0 / 0 |
| source/SpamArchive-2013 | v62 | high-recall | 108922 | 84.8258% | 0 / 0 |
| source/SpamArchive-2013 | v64 | conservative | 108922 | 71.5255% | 0 / 0 |
| source/SpamArchive-2013 | v64 | high-recall | 108922 | 83.0135% | 0 / 0 |
| source/SpamArchive-2014 | v62 | conservative | 95705 | 64.1983% | 0 / 0 |
| source/SpamArchive-2014 | v62 | high-recall | 95705 | 83.4648% | 0 / 0 |
| source/SpamArchive-2014 | v64 | conservative | 95705 | 69.3130% | 0 / 0 |
| source/SpamArchive-2014 | v64 | high-recall | 95705 | 80.2853% | 0 / 0 |
| source/SpamArchive-2015 | v62 | conservative | 109426 | 64.9480% | 0 / 0 |
| source/SpamArchive-2015 | v62 | high-recall | 109426 | 90.4474% | 0 / 0 |
| source/SpamArchive-2015 | v64 | conservative | 109426 | 75.3340% | 0 / 0 |
| source/SpamArchive-2015 | v64 | high-recall | 109426 | 88.8664% | 0 / 0 |
| source/SpamArchive-2016 | v62 | conservative | 159976 | 73.4223% | 0 / 0 |
| source/SpamArchive-2016 | v62 | high-recall | 159976 | 95.3180% | 0 / 0 |
| source/SpamArchive-2016 | v64 | conservative | 159976 | 83.5532% | 0 / 0 |
| source/SpamArchive-2016 | v64 | high-recall | 159976 | 93.6378% | 0 / 0 |

HAM here is newly selected after exclusion, not the old 75271-message replay. Source labels are preserved. A high aggregate score on spam-heavy archives cannot by itself establish transfer to customer traffic.
