# v84 source attribution and dataset limits

Only public email data were used; raw messages are not redistributed in model
artifacts or experimental model releases. This is a research experiment, not an
assertion that every source has the same license or complete provenance.

- Parent v82 corpus: reports/v79-frozen.json and reports/v80-extra-frozen.json.
  These preserve source counts, pinned revisions/hashes, label mappings and
  inherited limits. Individual sources include Enron-Spam, SpamAssassin,
  TREC05/06/07, CEAS08, Ling, GenSpam and labelled email2010.
- Public labelled email2010: realprogrammersusevim/email-dataset, revision
  84209612df4c55831f074a65b7c137a4b980059d:
  https://github.com/realprogrammersusevim/email-dataset .
- Seven-email source collection: JinqiangDing/seven-phishing-email-datasets,
  pinned revision deb4cc27dc2754c14cb157aace412727f77b41dd:
  https://huggingface.co/datasets/JinqiangDing/seven-phishing-email-datasets .
- GenSpam: Cambridge public research corpus:
  http://www.cl.cam.ac.uk/Research/NL/nl-download/GenSpam.tar.gz .
- Historical spam trap mail: Bruce Guenter SPAM Archive,
  https://untroubled.org/spam/ . Publisher grants unrestricted research/use;
  labels are trap-derived weak positives, not individually adjudicated.
  The2012 archive SHA256 is65bdb680fe95f3d2f52d24d99af7b7d6490c03fc5508313e8f5edefb4672de41.
- Nazario phishing archives: https://monkey.org/~jose/phishing/ . Exact year
  hashes and body-only split details are preserved in reports/v80-extra-frozen.json.
- Previously evaluated v64 transfer input: reports/v64-transfer-result.json
  preserves source URLs, source/archive hashes and the five input shard hashes.
  Prepared subject/body excerpts retain their existing labels and preprocessing
  limitations. They are no longer independent test material for v84.
- Merged-public source: locuoco/the-biggest-spam-ham-phish-email-dataset-300000,
  revision23029ecd499f6d2c564f5ddec4ac69964becce9c:
  https://huggingface.co/datasets/locuoco/the-biggest-spam-ham-phish-email-dataset-300000 .
  The prior experiment's incomplete original provenance is preserved, not resolved
  by repackaging it. Do not claim a universal license or independent-source proof.

Backbone: sentence-transformers/all-MiniLM-L6-v2, Apache2.0, revision
1110a243fdf4706b3f48f1d95db1a4f5529b4d41. Publisher license:
https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/LICENSE .
Exact per-file hashes accompany each checkpoint package. Backbone weights are
frozen and are not included in the adapter release.
