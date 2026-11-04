# Korean preview: read-only probe, 4 November 2026

This is a frozen Spanish-adapter inference experiment on Korean text. It does
not create or fine-tune a Korean model and is not a native Korean email test.

Source: [joohans/korean-phishing-email](https://huggingface.co/datasets/joohans/korean-phishing-email),
revision `b8af3cc78ee0c6b475819169053304842f952587`, Apache-2.0.
The pinned `data/korean_phishing_samples.jsonl` contains 30 short text records,
15 labelled1 and15 labelled0. The publisher describes Korean augmentation and
a sample preview. Native email provenance has not been established. Its English
TRAIN/TEST files were not used. A roadmap promising20,000 samples is not proof
that a downloadable native Korean corpus exists.

Frozen model: v83 Spanish campaign, step300, multilingual MiniLM12-layer
backbone with48 internal LoRA matrices. The checkpoint SHA256 and original
backbone manifest are checked before inference; checkpoint, adapters and frozen
parameters are verified unchanged afterward. No optimizer is created. Only
the supplied text enters the classifier. fastText ko>=0.8 accepted all30 texts.

At the predeclared threshold0.5: TP14, FN1, FP4, TN11. Recall14/15=93.33%,
FPR4/15=26.67%. These are descriptive counts on a tiny augmented preview.
They do not establish deployment performance, independent native-email transfer,
or suitability for training. No threshold was selected on these outcomes.
No ES TRAIN/SELECT/TEST contents or customer mail were opened.

The code is `tools/probe_korean_preview.py`; the aggregate report is
`reports/ko-preview/2026-11-04.json`. Raw preview text and per-record scores are
kept outside Git. Run locally on CPU with the pinned Spanish checkpoint from
[run41787777009](https://github.com/Kirusha-dk/MailGuard/actions/runs/41787777009):

```bash
python tools/fetch_v83_inputs.py --language es --backbone-only
python tools/probe_korean_preview.py \
  --checkpoint /path/to/es-step300/checkpoint.pt \
  --backbone .cache/v83/backbone \
  --lid /path/to/pinned/lid.176.ftz \
  --out .cache/ko-preview/new-output-directory
```

For a separate Korean email campaign, obtain sufficiently labelled real Korean
mail from both classes, document consent/provenance/licence and label mapping,
then freeze family-disjoint TRAIN/SELECT/TEST partitions. Do not relabel Korean
SMS, social conversations or augmented previews as native email. The small
preview does not meet the existing class-support requirements, so training_ready
remains false. No monthly KO task or duplicate RU/ES controller was created.

Other screened leads: [jmjmjm3/kor-smishing-message](https://huggingface.co/datasets/jmjmjm3/kor-smishing-message)
is SMS; [Visualization Technology and Deep-Learning for Multilingual Spam Message Detection](https://doi.org/10.3390/electronics12030582)
uses Korean smishing and conversational data. These are not verified native
email SPAM/HAM corpora for this task.
