# MailGuard v67 — Turkish to English translation (retrospective A/B)

**Previously evaluated v66 samples: exploratory replay, not new validation.**
No new classifier fitting or threshold selection; exact SHA-pinned v64 and v65 checkpoints.

Selected 384 label-blind Turkish messages (197 SPAM, 187 HAM).
Translation token truncation: 37; model Helsinki-NLP/opus-mt-tr-en at a01ed04b0bd6624fb4f699e3a8d8880a7004e00f.

| Model | Original Recall | Translated-only Recall | Original OR translated Recall | FP original / translated / OR |
|---|---:|---:|---:|---:|
| v64-conservative | 0.00% | 0.00% | 0.00% | 0 / 0 / 0 (of 187 HAM) |
| v64-high-recall | 0.00% | 0.00% | 0.00% | 0 / 0 / 0 (of 187 HAM) |
| v65-conservative-rescue | 0.00% | 0.51% | 0.51% | 0 / 0 / 0 (of 187 HAM) |

Paired changes (numbers are for investigation only):

- v64-conservative: {}
- v64-high-recall: {}
- v65-conservative-rescue: {'rescuedFN': 1}

**Do not select an OR policy or tune cutoffs using this v66 TEST replay.**
Further deployment requires development-only controls and a genuinely new independent bilingual SPAM/HAM test.
No customer mail was used or transmitted. Only aggregated metrics and model references are saved.
