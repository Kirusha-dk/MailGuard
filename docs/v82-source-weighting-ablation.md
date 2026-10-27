# Bounded source-weighting ablation, 27 October 2026

User-authorized separate experiment. Branch: `mailguard-v82-source-weighting-20261027`.
Parent: run 39332953339, artifact 12226237483, checkpoint500 SHA256
`a5cafb2fface8a35c672e075e41e63fe4551a786b536e4e41a23026299b63f68`.

Both arms restore the same adapters, AdamW, scheduler, CPU RNG and TRAIN cursor8000.
Control retains the monthly source/class inverse-frequency weights; candidate removes
those factors. Both retain HAM cost2 and global mean normalization. Each receives
exactly 500 optimizer steps, ending at step1000/cursor16000. Fork checkpoints carry
their own experiment/arm/parent metadata and never use the monthly artifact name.
Original trainer, neural implementation and cycle request remain unchanged.

Evaluation reads frozen SELECT only: parent, control and candidate at separately
fitted ~98% development recall thresholds and at predefined0.5; per-identity lost/gained
TP and removed/added FP, source and length breakdowns. Historical control counts
and threshold must reproduce. The known parent historical loss replay discrepancy
is recorded explicitly, not silently overwritten. No CAL or regression inference;
no customer mail, automatic promotion or independent quality claim.

Push of the experiment files on this exact branch launches a single CPU job.
Concurrency is shared with the monthly campaign. Output artifact:
`mailguard-v82-source-weighting-20261027-experiment`, retained45 days.
It contains fork checkpoints, reports, text-free score caches and comparison.json.
The workflow never pushes changes to the monthly branch. It is a one-off experiment,
not another scheduler. Code-triggered reruns are not automatic continuation: first
inspect existing run/artifact and recover fork state if any optimization occurred.

Local command with pinned dependencies, verified public inputs and exact downloaded
parent in `ablation-parent/`: `python -u tools/run_v82_weighting_ablation.py`.
Existing fork checkpoints are restored with their original1000-step target; completed
forks are evaluated without granting new steps. Actions restores the latest separate
experiment artifact before running; missing fork files in diagnostic artifacts are expected.

## Replay guard correction

Read-only run40473359489 reproduced historical TP/FP but recomputed the development
threshold as0.01019284874200821 instead of0.010192839428782463 (delta9.313225746e-9).
The original exact equality check conflated fitted-threshold replay with decision replay.
CPU diagnostic40474339661 reproduced518/518 sampled cached scores exactly in native,
ATEN AVX2 and combined ATEN/oneDNN AVX2 modes. This does not establish full score parity
or prove a hardware cause. No optimizer steps occurred in either diagnostic.

Before training, verify the pinned audit NPZ SHA256 and full SELECT identity/label order,
then require zero lost/gained TP and zero removed/added FP at the historical parent
threshold. Repeat that same per-identity guard on control1000 before training candidate.
Any changed individual decision still blocks the experiment. Recomputed thresholds,
max score deltas and log-loss differences are reported separately; the existing1e-9
log-loss replay criterion remains reported explicitly. Historical files are never rewritten.
Parent/control paired evaluations use historical locked thresholds; candidate uses its
SELECT-fitted development threshold. Fixed0.5 comparisons are also retained.
