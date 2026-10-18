# MailGuard v76 — frozen v75 neural versus v69 on new academic EN source

**First time scoring the 2024 study republisher: 9,000 fixed messages.**
No model fitting, threshold, ensemble blending, or source selection based on these TEST outcomes.

Source sniff07/phishing-dataset pinned revision c21f524c014a49626dc61322a5f7b3194ac5c7d1.
Labels: 4266 malicious, 4734 HAM. Prior overlaps excluded: {'exact_prior': 12786, 'prefix_prior': 816}.

| Profile | Model | Recall | TP / SPAM | FP / HAM | FPR |
|---|---|---:|---:|---:|---:|
| zero-cal-fp | v69 | 15.92% | 679/4266 | 5/4734 | 0.106% |
| zero-cal-fp | v75 | 27.31% | 1165/4266 | 50/4734 | 1.056% |
Paired deltas for zero-cal-fp: {'spam_new_only': 486, 'ham_both_negative': 4684, 'spam_both_negative': 3101, 'spam_both_positive': 679, 'ham_new_only': 45, 'ham_both_positive': 5}.
| research-0p1pct-cal-fpr | v69 | 27.40% | 1169/4266 | 51/4734 | 1.077% |
| research-0p1pct-cal-fpr | v75 | 27.26% | 1163/4266 | 39/4734 | 0.824% |
Paired deltas for research-0p1pct-cal-fpr: {'spam_both_positive': 1108, 'ham_both_negative': 4675, 'spam_both_negative': 3042, 'spam_new_only': 55, 'ham_both_positive': 31, 'ham_old_only': 20, 'spam_old_only': 61, 'ham_new_only': 8}.

2024 academic publisher mirror; original email origins, label quality and licensing NOT independently verified.
Prior exact and same-prefix overlaps screened versus all known v68/v70/v71/v73. Semantic family overlap is still possible.
English only, cleaned CSV text, not raw customer .eml or Rspamd metadata.
0.01% FP target NOT validated by 9k; do not tune with this challenge result.
Do not promote a model solely because this result is favorable. No raw emails were committed.
