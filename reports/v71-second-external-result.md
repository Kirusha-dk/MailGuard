# MailGuard v71 — 10,000 English mail, second first-use source

**Frozen preexisting v69 model, unmodified thresholds.**
This is one additional public source, not production validation.

Input source: realprogrammersusevim/email-dataset pinned to 84209612df4c55831f074a65b7c137a4b980059d.
Archive SHA256: c24e30ac874480ae4f9eca99be20505b123276709a86a5b9a23800a557c841a9.
Original eligible before exclusions: 17616.
Overlap exclusions: {'first160_character_overlap': 594, 'exact_content_overlap': 3261}.
Fixed sample: 10000 (4758 SPAM, 5242 HAM).

| Frozen v69 profile | Recall | TP / SPAM | FP / HAM | FPR |
|---|---:|---:|---:|---:|
| zero-cal-fp | 62.95% | 2995 / 4758 | 2 / 5242 | 0.038% |
| research-0p1pct-cal-fpr | 84.62% | 4026 / 4758 | 11 / 5242 | 0.210% |

**Important**: source data was ASCII cleaned; original providers and original mail licensing are not independently verified.
Same release may aggregate older public corpora; exact/prefix overlaps screened, semantic near-duplicates cannot be completely excluded.
This is one provider, not random external suppliers and not the customer's mail.
English only; no inference about Russian email quality.
No threshold tuning allowed after these first-use measurements.
Raw emails and individual predictions/identities are never committed or uploaded.
