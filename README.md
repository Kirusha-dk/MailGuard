> Текущая инженерная итерация: **v41, исправление проверки на тех же 50k письмах**.
> Код и ограничения: [docs/V41.md](docs/V41.md).
> [Запуски v41](https://github.com/Kirusha-dk/MailGuard/actions/workflows/v41-audited.yml).
> Старые команды public benchmark ниже относятся к прежнему прототипу; его shell-скрипты
> отсутствуют в этой версии репозитория. Для текущей проверки используйте workflow v41.

# MailGuard v0.4.5

Local C# prototype that adds a trainable risk model and human-review prioritization on top of Rspamd.

## Project targets

1. Raise **spam recall to at least 90%** on a genuinely held-out test set.
2. Keep false positives low and always report them next to recall.
3. Replace the customer's current random ~1% manual sample with a risk-ranked ~1% queue and measure how much more useful it is.
4. Keep message contents local; no external AI API is used.

A result is not called "90%" merely because overall accuracy is 90%. The primary target is spam recall on unseen mail.


## Docker-first запуск

Начиная с v0.4.5, для полного запуска MailGuard на компьютере достаточно Docker Desktop / Docker Engine с Compose. Устанавливать .NET и Python на хост не требуется.

Полная инструкция: [`DOCKER.md`](DOCKER.md).

Самый важный публичный benchmark запускается одной командой:

```bash
./tools/run_public_benchmark_docker.sh
```

## v0.4.5 changes

- reproducible Apache SpamAssassin public-corpus benchmark;
- deterministic train/test corpus policy;
- service file `cmds` and hidden files are excluded from mail datasets;
- a `learn-rspamd` command trains Rspamd Bayes on TRAIN only;
- public benchmark can remove exact/near train-test leakage before scoring;
- threshold optimization uses actual validation breakpoints instead of a coarse 0.01 grid;
- if protected Rspamd already violates the requested FP ceiling, MailGuard does not silently add more spam decisions;
- JSON + Markdown benchmark summaries;
- manual-review benchmark reports both full-stream and residual-after-Rspamd strategies;
- the 1% human-review budget is calculated from the whole stream, even when already-caught Rspamd spam is excluded from the candidate queue;
- GitHub Actions workflow can run the entire benchmark without a Mac/PC after the repository is connected to GitHub.

## Architecture

```text
raw message
   |
   +----> Rspamd /checkv2 ------------------+
   |                                        |
   +----> local MIME/text/header features --+--> local C# model
                                                |
                                      +---------+---------+
                                      |         |         |
                                    SPAM      NORMAL    UNSURE

manual QA path:
all daily mail -> risk score -> highest-risk ~1% -> human review
```

Rspamd spam decisions are protected: this prototype may add detections, but it does not downgrade a message Rspamd has already classified as spam.

## Requirements for an actual run

- .NET 8 SDK
- Docker + Docker Compose
- Python 3.10+
- `curl`

## Fastest full public benchmark

```bash
bash tools/run_public_benchmark.sh
```

That command:

1. downloads the official Apache SpamAssassin public corpus;
2. creates fixed train/test directories;
3. resets Rspamd/Redis state;
4. builds MailGuard and runs Rspamd;
5. trains **Rspamd Bayes only on train**;
6. trains MailGuard on the same development data;
7. evaluates both on untouched test messages;
8. removes detected train/test leakage before scoring;
9. compares random 1% manual review with MailGuard ranking;
10. writes machine-readable and human-readable reports.

Main result files:

```text
reports/benchmark.md
reports/benchmark-summary.json
reports/predictions.csv
reports/threshold-curve.csv
reports/review-benchmark.json
```

## Public benchmark split

TRAIN:

- `20030228_easy_ham` — 2500 ham
- `20030228_spam` — 500 spam

TEST:

- `20030228_easy_ham_2` — 1400 ham
- `20030228_hard_ham` — 250 difficult ham
- corrected `20050311_spam_2` — 1396 spam

Total prepared messages: **6046**. The corrected 2005 `spam_2` archive removes one message that the corpus maintainers identified as mislabeled.

The public-corpus source is:
`https://spamassassin.apache.org/old/publiccorpus/`

### Important limitation

This corpus is old. It is useful for a reproducible engineering benchmark and for finding bugs, but **it does not prove current production quality**. The final answer for the customer must be measured on a recent, representative, labeled customer dataset (or another modern dataset accepted by the customer).

The corpus maintainers also warn that old messages can produce misleading results with modern live DNS blocklists/fuzzy services. Therefore the included public-benchmark Rspamd profile disables external reputation modules (`rbl`, `fuzzy_check`, `spf`, `dkim`, `dmarc`) and measures a reproducible local baseline: static/content rules + trained Bayes. Production evaluation should use the customer's real Rspamd configuration separately.

## Individual commands

### Start the benchmark Rspamd stack

```bash
docker compose up -d
```

### Build/self-test

```bash
dotnet build MailGuard.sln -c Release
dotnet run --project src/MailGuard.SelfTest -c Release --no-build
```

### Prepare public corpus

```bash
python3 tools/prepare_spamassassin.py
```

### Train Rspamd Bayes — TRAIN only

```bash
dotnet run --project src/MailGuard.Cli -- learn-rspamd \
  --spam data/public/train/spam \
  --ham data/public/train/ham \
  --controller http://127.0.0.1:11334/
```

Never pass `data/public/test/...` to `learn-rspamd`.

### Train MailGuard

```bash
dotnet run --project src/MailGuard.Cli -- train \
  --spam data/public/train/spam \
  --ham data/public/train/ham \
  --model models/mailguard-public.mg \
  --rspamd http://127.0.0.1:11333/ \
  --max-fpr 0.005
```

### Evaluate held-out test

```bash
dotnet run --project src/MailGuard.Cli -- evaluate \
  --spam data/public/test/spam \
  --ham data/public/test/ham \
  --model models/mailguard-public.mg \
  --rspamd http://127.0.0.1:11333/ \
  --drop-leakage true
```

### Compare random 1% vs risk-ranked 1%

```bash
dotnet run --project src/MailGuard.Cli -- review-benchmark \
  --spam data/public/test/spam \
  --ham data/public/test/ham \
  --model models/mailguard-public.mg \
  --percent 1 \
  --runs 2000 \
  --output reports/review-benchmark.json
```

It prints two comparisons:

- **full stream** — direct random 1% vs risk-ranked 1%;
- **residual after Rspamd** — the same total-stream review budget, spent only on messages Rspamd did not already catch as spam.

## Rspamd cache

Repeated scans are cached under `.cache/rspamd`. Clear the cache or change `--cache-namespace` whenever the Rspamd version, configuration, rules or Bayes state changes.

## What counts as success

For the public benchmark and later customer benchmark, report at least:

- `spam detected / spam total`;
- spam recall;
- `false positives / ham total`;
- false-positive rate;
- delta versus Rspamd;
- manual-review lift versus random 1%;
- exact size/source/date of the test population.

The customer result is successful only when the agreed target is reached on unseen representative mail, not when a threshold has been tuned on the same test set.

## GitHub Actions: run without a local computer

v0.4.5 includes an automatic public benchmark workflow. A push to `main`/`master` starts the full Docker benchmark on a GitHub-hosted Ubuntu runner. The final `Bayes vs Bayes + MailGuard` table is written into the Actions job summary, and all evidence is uploaded as an artifact. See `GITHUB-ACTIONS.md`.
