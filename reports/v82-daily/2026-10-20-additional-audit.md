# v82 development audit — 2026-10-20 (additional check)

Source: reports/v82-daily/2026-10-20.json; workflow run 39329856078; replay verification run 39332953339.

- Session completed: 500 optimizer steps, global_optimizer_steps=500, cursor=8000, completed_sessions=1.
- Internal LoRA adapter tensors changed: 24; delta L2=4.4469943; trainable parameters=74113.
- Checkpoint artifact: mailguard-v82-english-20261020-state, verified present on first run, not expired at audit time.
- Replay verification restored checkpoint and completed with invocation_optimizer_steps=0. No extra training on same date.
- SELECT (development, **not independent**): TP=10378/10589 spam, recall=98.007%; FP=2888/12201 HAM, FPR=23.6702% (236702 per million); threshold=0.01019284 selected on this same SELECT.
- This threshold does not satisfy the 10 FP/million aspiration; observed FPR is approximately 23,670 times that target. No independent million-HAM test was run.
- Source-level TP/FP counts are absent from this day's published report; do not infer source transfer or claim a source breakdown.
- No evidence of production improvement, no candidate promotion. v80 remains unchanged.

Next bounded work: verify that future development reports include source-level TP/FP at one shared threshold; inspect candidate errors only on TRAIN/SELECT and preserve untouched CAL/holdout boundaries. Do not use old 100k or Nazario-2024 labels to choose thresholds.

No further optimizer steps or workflow trigger requested for 2026-10-20.
