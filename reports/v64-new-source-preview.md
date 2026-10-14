# MailGuard: ready new labelled-source cohort

Previously unused labelled public-source cohort after prior-source exclusions. GenSpam is anonymized; published ASCII mail is from a known Enron-like domain with removed headers; Nazario is one manually classified inbox. This cohort will also be included in the pending larger comparison with the same frozen models; it is not a second independent test. Exclusions include short bodies and duplicates. Source labels are preserved; results do not establish customer quality.

Actual count: 25971; SPAM: 15234; HAM: 10737. This is not the 500000-message result.

| Group | Version | Profile | Spam recall | FP / HAM | FPR | Wilson 95% upper |
|---|---|---|---:|---:|---:|---:|
| all | v62 | conservative | 5.0151% | 0 / 10737 | 0.0000% | 0.0358% |
| all | v62 | high-recall | 57.9625% | 88 / 10737 | 0.8196% | 1.0086% |
| all | v64 | conservative | 10.7720% | 0 / 10737 | 0.0000% | 0.0358% |
| all | v64 | high-recall | 38.9523% | 7 / 10737 | 0.0652% | 0.1345% |
| new-mail-known-domain | v62 | conservative | 13.6765% | 0 / 2422 | 0.0000% | 0.1584% |
| new-mail-known-domain | v62 | high-recall | 81.8382% | 3 / 2422 | 0.1239% | 0.3636% |
| new-mail-known-domain | v64 | conservative | 24.6569% | 0 / 2422 | 0.0000% | 0.1584% |
| new-mail-known-domain | v64 | high-recall | 62.2549% | 1 / 2422 | 0.0413% | 0.2335% |
| source/GenSpam | v62 | conservative | 1.8231% | 0 / 8315 | 0.0000% | 0.0462% |
| source/GenSpam | v62 | high-recall | 49.6977% | 85 / 8315 | 1.0222% | 1.2622% |
| source/GenSpam | v64 | conservative | 5.3390% | 0 / 8315 | 0.0000% | 0.0462% |
| source/GenSpam | v64 | high-recall | 30.2577% | 6 / 8315 | 0.0722% | 0.1574% |
| source/Nazario-2025 | v62 | conservative | 2.4814% | 0 / 0 | n/a | n/a |
| source/Nazario-2025 | v62 | high-recall | 36.7246% | 0 / 0 | n/a | n/a |
| source/Nazario-2025 | v64 | conservative | 15.1365% | 0 / 0 | n/a | n/a |
| source/Nazario-2025 | v64 | high-recall | 34.9876% | 0 / 0 | n/a | n/a |
| source/Published-ASCII-mail | v62 | conservative | 13.6765% | 0 / 2422 | 0.0000% | 0.1584% |
| source/Published-ASCII-mail | v62 | high-recall | 81.8382% | 3 / 2422 | 0.1239% | 0.3636% |
| source/Published-ASCII-mail | v64 | conservative | 24.6569% | 0 / 2422 | 0.0000% | 0.1584% |
| source/Published-ASCII-mail | v64 | high-recall | 62.2549% | 1 / 2422 | 0.0413% | 0.2335% |
| unseen-source | v62 | conservative | 1.8469% | 0 / 8315 | 0.0000% | 0.0462% |
| unseen-source | v62 | high-recall | 49.2290% | 85 / 8315 | 1.0222% | 1.2622% |
| unseen-source | v64 | conservative | 5.6930% | 0 / 8315 | 0.0000% | 0.0462% |
| unseen-source | v64 | high-recall | 30.4285% | 6 / 8315 | 0.0722% | 0.1574% |
