# MailGuard v62: large gold HAM test

Historical public gold labels, not customer mail. All TRAIN/SELECT/CAL identities and campaign families are excluded, including legacy-decoding aliases. Exact identity, body and normalized-template duplicates and contradictory labels are excluded. Sources have been inspected during engineering; additional messages are not an independent production benchmark. Thread relatives may remain. No fitting or threshold selection occurs in this evaluation; HAM-only results cannot establish spam recall.

| Group | Profile | HAM | FP | FPR | 95% FPR upper bound |
|---|---|---:|---:|---:|---:|
| additional | conservative | 47258 | 0 | 0.00000% | 0.00813% |
| additional | high-recall | 47258 | 113 | 0.23911% | 0.28738% |
| all | conservative | 75271 | 0 | 0.00000% | 0.00510% |
| all | high-recall | 75271 | 164 | 0.21788% | 0.25383% |
| previously-scored | conservative | 28013 | 0 | 0.00000% | 0.01371% |
| previously-scored | high-recall | 28013 | 51 | 0.18206% | 0.23927% |
| script/cjk | conservative | 2364 | 0 | 0.00000% | 0.16223% |
| script/cjk | high-recall | 2364 | 4 | 0.16920% | 0.43428% |
| script/latin | conservative | 72467 | 0 | 0.00000% | 0.00530% |
| script/latin | high-recall | 72467 | 160 | 0.22079% | 0.25770% |
| script/other | conservative | 440 | 0 | 0.00000% | 0.86550% |
| script/other | high-recall | 440 | 0 | 0.00000% | 0.86550% |
| source/Assassin | conservative | 1527 | 0 | 0.00000% | 0.25094% |
| source/Assassin | high-recall | 1527 | 3 | 0.19646% | 0.57604% |
| source/CEAS-08 | conservative | 13461 | 0 | 0.00000% | 0.02853% |
| source/CEAS-08 | high-recall | 13461 | 62 | 0.46059% | 0.58996% |
| source/Enron | conservative | 8304 | 0 | 0.00000% | 0.04624% |
| source/Enron | high-recall | 8304 | 8 | 0.09634% | 0.19000% |
| source/Ling | conservative | 589 | 0 | 0.00000% | 0.64797% |
| source/Ling | high-recall | 589 | 0 | 0.00000% | 0.64797% |
| source/TREC-05 | conservative | 25679 | 0 | 0.00000% | 0.01496% |
| source/TREC-05 | high-recall | 25679 | 65 | 0.25313% | 0.32247% |
| source/TREC-06 | conservative | 6642 | 0 | 0.00000% | 0.05780% |
| source/TREC-06 | high-recall | 6642 | 8 | 0.12045% | 0.23751% |
| source/TREC-07 | conservative | 15724 | 0 | 0.00000% | 0.02442% |
| source/TREC-07 | high-recall | 15724 | 14 | 0.08904% | 0.14941% |
| source/TREC06-Chinese-original | conservative | 3345 | 0 | 0.00000% | 0.11471% |
| source/TREC06-Chinese-original | high-recall | 3345 | 4 | 0.11958% | 0.30709% |
