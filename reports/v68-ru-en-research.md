# MailGuard: Russian + English research protocol (15 October 2026)

**Objective:** improve generalization to Russian and English emails, ideally reaching >=90% spam recall with extremely few false positives. **No guarantee** exists for arbitrary 10,000 emails or previously unseen suppliers. This document is a research protocol, not an acceptance result.

## Prior measured evidence

- v64 on its first-use 500k corpus: English/Latin-script spam recall 33.57% (conservative), 61.58% (higher recall); 7/10,737 HAM false positives for the latter. Aggregate multilingual scores are dominated by Chinese spam archive and are inappropriate as evidence for English.
- v66 first-use on 1,307 candidates: 35.45% overall recall for v64 high-recall, 0 FP on 489 HAM; Turkish stemmed corpus 0% on all versions, and 323 public phishing emails supplied the recognized true positives.
- v67 translation of a 384-item subset: 1 extra detection of 197 spam emails, no FP on 187 HAM. This subset had previously been evaluated, so results are retrospective.

Evidence in [v64](v64-transfer-result.md), [v66](v66-firstuse-result.md), [v67](v67-translation-result.md).

## Researched candidate corpora

- [alt-gnome/telegram-spam](https://huggingface.co/datasets/alt-gnome/telegram-spam): 21,411 annotated Russian Telegram messages, CC0-1.0. Accessible for small-scale Russian-language **training** but not representative of full MIME mail, different ham distribution, limited provenance. Must NEVER call Telegram model performance independent Russian **email** quality.
- [JinqiangDing/seven-phishing-email-datasets](https://huggingface.co/datasets/JinqiangDing/seven-phishing-email-datasets): 203,017 English mail records across Enron, CEAS, Ling, SpamAssassin and TREC. Its distributor does not establish one universal license for the compiled data. These email sources overlap previous MailGuard development, so grouped holdouts are engineering/regression measurements and not first-use across sources.
- [ruSpamModels/russian-spam-detection](https://huggingface.co/datasets/ruSpamModels/russian-spam-detection): Russian data gated behind acceptance and CC-BY-NC restrictions; unsuitable for unattended download or customer commercial use without permission.
- [AugustLight/telegram_spam_ru](https://huggingface.co/datasets/AugustLight/telegram_spam_ru): 1,830 Russian chat messages, gated with noncommercial terms; not suitable as public original-mail TEST.

No source with both representative raw Russian emails and reliably adjudicated spam/ham labels was verified. No customer email has been accessed.

## v68 experiment (bounded CPU research)

- Train independent hashed word/character classifiers for RU and EN, rather than translating all incoming text or changing frozen v64/v65.
- SHA256-pinned publisher revisions and content grouping, exact duplicate rejection and conflicting-label exclusion. TRAIN, CAL and 5,000 evaluation rows per language are chosen by deterministic campaign-family hashes and identity order without using target metrics.
- TRAIN caps: 18,000 RU and 70,000 EN; two SGD epochs; two predeclared calibration policies per language. The conservative policy requires 0 FP on CAL; the exploratory alternative tolerates approximately 0.1% CAL FPR, which **does not** meet the desired 0.01%.
- Save model and thresholds **before** reading any TEST labels, produce confusion matrices and Wilson confidence bounds. Keep model artifact separately; do not replace stable v64 or v65.
- The 10,000-record exercise is explicitly **within-source**, with RU chat rather than email. It can expose modelling/engineering regressions but cannot demonstrate generalization to *any 10,000 emails*.

## Actual acceptance criteria (future)

1. Freeze the model before finding a new independent RU/EN mail source. Avoid all previously inspected source pools and exact/family overlaps; record hashes and original adjudicated labels.
2. Evaluate several distinct source-specific 10,000-mail cohorts, not just a favourable aggregate, and publish per-source and per-language TP/FN/FP/TN.
3. Aim for >=90% spam recall on each relevant independent source. Prove FPR using a much larger independent HAM pool: at 0 observed FP, roughly 30,000 HAM provide a one-sided 95% upper bound of 0.01% (under IID/binomial assumptions).
4. If external rights are unclear or no representative Russian mail exists, report *not validated*; do not fabricate independent test evidence or send customer data without permission.

## High-priority request to customer

Ask Svyatoslav about the proportions of Russian/English in **both spam and legitimate emails**, typical MIME HTML/text format, and whether privacy-safe, independently labelled audit samples can be supplied with explicit consent. This determines the true target domain.
