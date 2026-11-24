# English recovery after v86 regression — 24 November 2026

The user requested repair after an English regression evaluation. The existing
v86 controller is paused, not replaced by a second schedule. v82, v84, other
languages, frozen corpora, PR12 and the original v86 checkpoint remain intact.
No product promotion is authorized by this experiment.

The retained parent is v84 step46000, SHA256
2e62947281989573c47e322e2dd372a9d0690750f4490c9238f9f962ef17e685.
The paused v86 checkpoint is step49500, SHA256
56e8284f958dbe59bc7c90c590dbc3458d72fdaae31a4e6b78e6d67bd88a5b47.
Both were downloaded and verified in durable v86 bootstrap/weekly packages.

Seven completed development sessions did not improve fitted98 FP from548;
latest FP806, while fixed-threshold TP fell from10394 to9733. The original
stop condition exempted epoch0. The fix applies patience from the first
completed session, including a zero-gradient preflight for already stalled
states. Best-adapter replacement additionally requires no regression in
fixed-threshold TP/FP, fitted98 FP and development loss. Existing metadata
CONFIG, optimizer/scheduler and frozen dataset definitions are unchanged.

A separate, unscheduled CPU repair pilot is capped at64 optimizer steps and
1024 different TRAIN rows:256 old SPAM,256 new SPAM,512 HAM. Old SPAM replay
was absent from v86's support set. Rows preserve original labels. Adam and RNG
start from v86 step49500; a new lower-LR repair scheduler is explicitly planned
once. The original scheduler, cursor, full exposure ledgers and provenance are
preserved. Old-SPAM counts come from the pinned parent; other counts come from
v86. No row reaches more than3 lifetime gradient presentations. The final8000
v86 rows of24 November are excluded from this separately authorized pilot.

The objective uses equal-class hard-label BCE and teacher regularization on
old TRAIN replay (retained parent, T=2). HAM-only cost/error boosts are removed
for this pilot. All24 LoRA matrices/head can update; the pinned backbone stays
frozen. No benchmark row, customer mail, SMS/chat, translation or synthetic
mail enters training. This does not reset v86's daily budget or create a month.

The candidate is selected only on the old22790-row SELECT development, never
on the20630-row English regression benchmark. If the strict gate fails, the
exact retained parent adapter is selected and the unsuccessful candidate is
kept separately. The selected inference adapter and the repair optimizer
checkpoint are explicitly different outputs; no parent weights are paired
with a misleading v86 optimizer for continuation. Reusing the public benchmark
now supplies regression evidence, not a fresh independent test. No98% transfer
or10ppm guarantee is made.

Data attribution and source conditions remain in docs/v84-source-attribution.md,
docs/v86-english-month.md and research/v86. Trap labels are weak; old
Merged-public provenance is incomplete. Raw text stays in local/runner cache.
Only code, provenance and small verified aggregates belong in Git; model state
belongs in separate experimental artifacts/packages. No cron or new manager.
