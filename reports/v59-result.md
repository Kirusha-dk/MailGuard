# MailGuard v59 — new public mail

New model and thresholds were locked before opening TEST. No old TEST errors were added to training.

| Model/profile | Mail | Ham | Spam recall | FP | FN | FP / ham |
|---|---:|---:|---:|---:|---:|---:|
| v57 conservative | 11831 | 1968 | 18.79% | 1573 | 8010 | 79.9289% |
| v59 conservative | 11831 | 1968 | 47.39% | 0 | 5189 | 0.0000% |
| v57 high-recall | 11831 | 1968 | 26.48% | 1776 | 7251 | 90.2439% |
| v59 high-recall | 11831 | 1968 | 66.13% | 1 | 3341 | 0.0508% |

Reused English/CSDMC engineering regression (not independent):

| Model/profile | Mail | Ham | Spam recall | FP | FN |
|---|---:|---:|---:|---:|---:|
| v59 conservative | 44615 | 24017 | 67.61% | 15 | 6672 |
| v57 conservative | 44615 | 24017 | 92.27% | 46 | 1593 |
| v59 high-recall | 44615 | 24017 | 92.01% | 58 | 1645 |
| v57 high-recall | 44615 | 24017 | 92.89% | 51 | 1465 |

Public historical Chinese mail is not a customer sample. Spam traps are weak spam labels. Family fingerprints cannot exclude every semantic near-duplicate. Unselected old SpamArchive pool is unavailable; all selected 500k, whole HF source pools and frozen 110k are blocked.

Weak Japanese ham is training-only. This result does not validate customer production quality.
