# v84 English replay: prepared experiment, not activated

User-authorized design: expand the English corpus, reuse previously evaluated
public emails (especially errors), target up to50,000 distinct rows per date,
and allow up to six calendar months. Proposed bounded window5 November2026
through4 May2027 inclusive. Early stopping and finite exposure limits can stop
much earlier; the duration is not a promise of six months of useful training.

Current stage is **prepare**, not completed training or an active daily service.
No new controller/cron/task has been created. The latest RU/ES/TR instruction
preserves the original v82 request/schedule/weights and prohibits new tasks;
activation of an English replay controller is therefore left pending. The
existing v82 monthly campaign remains unchanged. This branch is a separate
experiment, not a modification to its checkpoint metadata or original schedule.

## Data conversion and honest evaluation

The source is the pinned v82 corpus plus previously scored public CAL/TEST
partitions and English rows from the five pinned v64 transfer shards. The old
v64 material is normalized subject/body excerpts, not full original EML. Its
500,000 total rows include many non-English messages; those must not be counted
as500,000 English training examples. Actual retained counts are emitted by the
freeze and must be committed as research/v84/frozen.json before training.

The640 known false negatives from the4 November30k spam-only audit are recovered
from the SHA256-pinned original2012 archive and verified against their original
raw hashes, canonical identities and families. They inherit weak spam-trap
labels, not new human-adjudicated labels. No HAM labels are invented. That30k
audit and the other consumed CAL/TEST pools are retired from independent
evaluation for v84. Future scores there are training/development/regression
observations. The original22,790 SELECT examples stay isolated from gradients;
identity, prefix-family and additional20-word-window checks screen overlaps.
Cross-label conflicting families are excluded. Complete semantic independence
and provenance of the preexisting merged source are not established.

No new independent test has been established. Production promotion and claims
of98% recall or10ppm are not authorized by this protocol. All evaluation uses
an explicit development label and includes TP/FP/FN/TN and the HAM denominator.
Private/customer messages are prohibited in Git/Actions/external AI APIs.

## State, budget and exposure accounting

Initial parent: v82 completed step8500 from run41973548171, checkpoint SHA256
7be6f93c074e4460941ba444abba486b23cd601d3a21bebe130c1723a6358784.
Pinned backbone all-MiniLM-L6-v2 revision1110a243fdf4706b3f48f1d95db1a4f5529b4d41.
The24 internal LoRA matrices and head are inherited, along with Adam moments
and RNG. The parent checkpoint is immutable. The expanded dataset requires an
explicit new cursor/order and linear scheduler starting at the parent's current
learning rate. This scheduler migration is recorded, not passed off as an
unchanged v82 continuation. Full v84 interruptions restore optimizer, scheduler,
RNG, order, cursor, per-row exposure counts and the original daily target.

Maximum daily target:50,000 distinct canonical rows, batch16,3,125 optimizer
steps. Inherited v82 row exposure counts (one or two) are retained. A canonical
row is presented at most3 times including those inherited exposures; known
errors are not repeated indefinitely. Known errors are ordered early in each
pass. Wrong predictions within training batches receive a factor2 loss weight;
known historical false negatives receive an additional factor1.5, HAM a factor2
base weight, and source/class imbalance receives bounded square-root weighting.
These are new experimental objectives, not established improvements.

The existing SELECT threshold from parent step8500,0.04897100105881691, defines
online training error weighting and a fixed development comparison. A separate
development-fitted recall98 threshold is reported explicitly as fitted. Best
adapters are chosen by development FP/log-loss; early stop after5 non-improving
completed sessions following an epoch, or at the exposure/pass cap. Never
repeat a completed date, request a future date or silently restart a missing
checkpoint. CPU-only invocations are bounded, not continuous services.

## Prepared workflow and remaining activation work

The draft workflow uses a draft PR into the existing external-experiment branch
to restore its exact public reference-data cache. It has no schedule trigger.
Stage prepare freezes only data; stage train requires byte-equivalent committed
manifest. A64-step checkpoint milestone and the remaining daily work share the
same original50k/3125 budget. Artifacts contain only state/weights/aggregates;
normalized public training text is confined to the runner cache.

Eight engineering tests pass: exact interrupted optimizer/scheduler/RNG/cursor
restoration, fixed-parameter preservation, original-target preservation, date
and rollback guards,50k scheduling uniqueness, known-error prioritization,
selection/conflict exclusion, HAM metric denominators and byte-equivalent email-normalization fast paths. Invented fixtures in
these tests do not measure email quality. Real expanded-corpus training and
six-month operation are **not yet verified**.

Before activation: commit and reproduce the actual frozen manifest, verify the
runner cache/CPU throughput and real optimizer milestone, establish durable
parent/bootstrap storage before source artifact expiry, and resolve the
controller scope against the latest instruction preserving v82/no new tasks.
Do not claim daily50k training or a million-row corpus until actually confirmed.
