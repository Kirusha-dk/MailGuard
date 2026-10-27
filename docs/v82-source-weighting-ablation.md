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
forks are evaluated without granting new steps. Ordinary Actions retries must restore
their experiment artifact first if either fork has already trained.
