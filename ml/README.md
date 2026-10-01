# MailGuard neural classifier

This folder contains the deployable neural classifier. It is separate from the public benchmark scripts.

## Dataset layout

```text
data/
  train/
    spam/
    ham/
  val/
    spam/
    ham/
  test/
    spam/
    ham/
```

Each folder contains raw email files such as `.eml`. Message contents stay local.

## Train

With Docker and the repository Rspamd stack:

```bash
docker compose up -d redis rspamd
docker compose run --rm mailguard-train
```

The default output is `models/mailguard-v7.pt`.

Training uses only messages that Rspamd did not already protect as spam. The validation set chooses the spam threshold. By default the allowed added false-positive rate is 0.01% of validation ham; for small validation sets this usually means zero additional false positives.

## Evaluate on untouched data

```bash
docker compose run --rm mailguard-evaluate
```

This writes `reports/evaluate-v7.json`.

Do not tune the model or threshold on the test folder.

## Classify one message

Put an email at `data/inbox/example.eml`, then run:

```bash
docker compose run --rm mailguard-predict --input /workspace/data/inbox/example.eml
```

The result contains the neural spam probability, the Rspamd result and the final SPAM/UNSURE decision.

## Important

The public SpamAssassin corpus is useful for development but is old. Production quality must be measured on recent representative mail. Private mail should be trained and evaluated on the customer's own machine/server, not uploaded to GitHub Actions.
