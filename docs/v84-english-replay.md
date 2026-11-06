# v84 English replay: separately authorized bounded campaign

User-authorized design: expand the English corpus, reuse previously evaluated
public emails (especially errors), target up to50,000 distinct rows per date,
and allow up to six calendar months. Proposed bounded window5 November2026
through4 May2027 inclusive. Early stopping and finite exposure limits can stop
much earlier; the duration is not a promise of six months of useful training.

Activation authorized5 November2026. A separate daily v84 controller runs within
the bounded window while original v82 and the RU/ES/TR controller remain unchanged.
Only one v84 controller is permitted. The initial training invocation must be
verified from its actual Actions checkpoint; activation is not proof that a
50,000-row session completed.

Frozen corpus:358,657 TRAIN (223,550 SPAM/135,107 HAM),22,790 SELECT.
All640 recovered original false negatives are retained. This is an expansion
from90,680 v82 TRAIN rows, not a million-row corpus. New independent test
quality has not been measured.

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

## Activation and verification

The draft workflow uses a draft PR into the existing external-experiment branch
to restore its exact public reference-data cache. It has no schedule trigger.
Stage prepare freezes only data; stage train requires byte-equivalent committed
manifest. A64-step checkpoint milestone and the remaining daily work share the
same original50k/3125 budget. Artifacts contain only state/weights/aggregates;
normalized public training text is confined to the runner cache.

Nine engineering tests pass: exact interrupted optimizer/scheduler/RNG/cursor
restoration, fixed-parameter preservation, original-target preservation, date
and rollback guards,50k scheduling uniqueness, known-error prioritization,
selection/conflict exclusion, HAM metric denominators, tied-score FP-budget handling and byte-equivalent email-normalization fast paths. Invented fixtures in
these tests do not measure email quality. Actual optimizer progress and daily results must be verified from Actions.
A six-month date window is not six months of completed training.

The frozen manifest is committed. Bootstrap parent/audit metadata are preserved
as a prerelease before training. Corpus caches must match the frozen manifest;
if both cache and source artifacts become unavailable, stop and report missing
inputs rather than change data or restart weights. Verify actual runner progress
and throughput, and preserve final state/best adapters/metadata at any early stop
or the end of the window. Do not claim daily50k completion until confirmed.

## Activated controller

One enabled controller: Automation_597c9b42f8388191b72b0e7e3bcd5ae1.
Morning flexible schedule08:00–09:00 Europe/Moscow,6 November2026 through
4 May2027 inclusive,180 scheduled occurrences following the initial5 November
session. Stop earlier at exposure/epoch/early-stop limits and preserve final
experimental package; do not automatically renew the campaign.

Execution PR: https://github.com/Kirusha-dk/MailGuard/pull/10 .
Initial run: https://github.com/Kirusha-dk/MailGuard/actions/runs/42025999444 .
Each run's actual checkpoint, rows and optimizer steps are authoritative.
Use [skip ci] for documentation-only bookkeeping commits on this PR: a PR path
filter can otherwise see the whole PR diff and enqueue an unwanted replay.
The controller changes cycle.json only when it intends a bounded training run.
Source attribution accompanies packages: [source notes](v84-source-attribution.md).
