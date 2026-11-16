# v85: isolated English error-focused pilot

User authorized additional English datasets and prioritized classifier mistakes.
This is one bounded100-optimizer-step CPU experiment from the completed v84
checkpoint42875 (16 November), SHA256
3b77371c3d413a7dc90f66ae69d0744c93e1b5a4f17631e9517b51d5878e0f79.
The v84 campaign, data freeze, request, scheduler, weights and controller remain
unchanged. No new schedule/controller or automatic product promotion is created.

Pinned public sources:
- CSDMC2010 processed mirror: https://github.com/zrz1996/Spam-Email-Classifier-DataSet
  revision36b5435af66a037a5cd474874950ca38c837e953. Original competition source:
  http://csmining.org/index.php/spam-email-datasets-.html . The directory labels
  HAM0/SPAM1 are inherited. The pinned mirror contains2949 HAM/1378 SPAM,
  while its README says2929 HAM; no label/count is invented to fix that discrepancy.
  The mirror concatenates subject and body and sometimes emits a generated MIME
  envelope. That wrapper is removed; sender/date/file/class names are not features.
  Repository MIT covers software; upstream mail-text rights are not separately
  established. Do not describe all mail as MIT-licensed.
- IWSPA-AP2018 official-training mirror: https://github.com/BarathiGanesh-HB/IWSPA-AP
  revisiond854f44ec2f26d8110bed9450869ac97e19db754,
  data/IWSPA-AP-traindata.zip. Native English legitimate/phishing body text,
  explicit directories legit0/phish1; inherited labels, no relabelling.
  No standard data licence was found in the mirror. Preserve this uncertainty
  and cite the IWSPA-AP2018 shared task and publisher; no raw redistribution.
  Do not use the publisher's unlabelled test archive as invented HAM/SPAM.

Source ZIP SHA256 and exact retained fingerprints/counts are in research/v85/frozen.json.
Only native email is accepted, fastText English>=0.8; no SMS, chat, translated or
synthetic mail. CSDMC spam and IWSPA phishing are different positive-class scopes;
this pilot does not establish a universal modern spam filter. Customer mail is
not used. Raw/normalized text remains in runner cache, never Git or model packages.

All v84 TRAIN358657 and SELECT22790 rows are excluded by canonical identity,
45-word prefix family and shared20-word windows in the first120 words. Quote/prefix
connected components of the additional mail are split before model scoring.
Frozen new split: TRAIN2637 (352 positive/2285 HAM), SELECT699 (89/610),
TEST630 (65/565). Semantic independence is not fully established; do not call
public-source labels individually adjudicated or ignore source/domain skew.

Mine mistakes only on the new TRAIN at fixed parent threshold0.04897100105881691:
false-positive HAM and false-negative positive messages. Deterministic order gives
priority to errors (up to3 canonical presentations) and retains correct support
(up to1). In each12-error/4-support block, available pools determine the actual
ratio; report measured counts, not a promised75%. Total<=100 steps /1600 row
presentations. A small/exhausted pool yields fewer steps; no endless retraining.
Error lists contain no SELECT/TEST rows. TEST contents are never opened by trainer.

Initialize LoRA/head and exact Adam/RNG from own English parent; record one explicit
new100-step scheduler for the new objective. Then exact own-pilot resume restores
Adam/scheduler/RNG/order/cursor/exposures. Missing pilot state after an attempted
run forbids a cold restart. Completed retry has0 new invocation steps.24 LoRA
matrices,74113 trainable parameters; pinned all-MiniLM-L6-v2
revision1110a243fdf4706b3f48f1d95db1a4f5529b4d41, frozen backbone.

Before/after on SELECT are development with TP/FP/FN/TN, recall, FP/HAM and FPR.
At89 positives, fitted98 is suppressed as insufficient support. Exploratory
HAM-budget thresholds are fitted on that same SELECT and not independent proof.
No10ppm claim, no automatic model replacement. Preserve candidate AND parent,
checkpoint, optimizer/RNG/scheduler, baselines, mining counts, exact backbone
manifest, frozen metadata and this attribution in an experimental prerelease.
Three engineering tests cover error selection, correct support, determinism and
canonical exposure cap, small-HAM FP budgets and insufficient fitted98 support. Their fixtures are not training data or quality evidence.
