# Million-scale English email source audit —16 November2026

Request: find one million additional real English emails for spam/ham training,
with emphasis on mistakes made by the current classifier. **Million-scale sources
were found; one million new, unique, labelled English training rows have not been
established. No new model training or schedule changes were performed.**

## Enron/FERC:1,226,178 messages reported by the publisher

[Native PST release](https://huggingface.co/datasets/intellekthq/enron-ferc-pst/tree/1c6490b25d3a16b71a070cbbef87d8fe394c3df3),
revision `1c6490b25d3a16b71a070cbbef87d8fe394c3df3`, declares English and1,226,178
messages in171 PST files,83,083,881,472 bytes (~83.1GB). The publisher retains
ordinary business correspondence, **spam and newsletters**, so the whole corpus
must not be labelled HAM. This is not the smaller Enron-Spam labelled subset.

Verified the171-file catalogue, every published SHA256 against Hugging Face LFS
content IDs, and downloaded the smallest PST (10,937,344 bytes): its checksum
matched and its PST signature was valid. The full1,226,178 count is the
publisher's count, not a fresh independent recount. Global canonical uniqueness,
English filtering and overlap with previous MailGuard Enron data remain unchecked.
No SPAM/HAM target annotation is supplied. It is therefore a candidate source,
not a ready supervised million-row corpus.

The release documents CC BY3.0 US, source attribution to ZL Technologies Inc.,
and Intellekt's modifications. Preserve its privacy/provenance limitations;
these do not establish complete de-identification or a blanket HAM label.
The full83GB was not downloaded into a workspace with about2.8GB free.

## Spam Archive2017:1,400,401 email files independently inventoried

[Archive](https://untroubled.org/spam/2017.7z),272,136,279 compressed bytes,
SHA256 `9744cc3fb7ad3cee1cf036a13a0df313a3af8deeda294da2813755d44eb4593d`.
Downloaded the archive and independently read its member inventory:
**1,400,401 files**,3,067,373,260 uncompressed bytes. The publisher describes
mail collected by spam-trap addresses and grants unrestricted use at
https://untroubled.org/spam/ . These are inherited weak positive labels, not
individually adjudicated mail; no HAM is provided.

Actually parsed the first5,000 archive members using MailGuard normalization
and the pinned fastText English>=0.8 gate:167 English examples,129 canonical
unique English examples /110 prefix families;4,078 non-English or uncertain
and755 short examples were excluded. This chronological prefix is **not a
representative random sample**; neither its language fraction nor its dedup rate
is extrapolated to the whole archive. The archive is not1.4 million verified
unique English emails. Full language filtering, dedup and old-data exclusion
remain outstanding. The129 sample examples were not added to TRAIN.

## Other misleading scale counts excluded

- Gmane mirror reports1,917,151 rows with authorIDs/fullText lists, but gives no
  SPAM/HAM labels or validated canonical email count. Do not label it all HAM.
- PhishingEmailDetectionv2.0 reports200,000 entries, of which177,356 are URLs
  and only22,644 are email samples. URLs are not emails.
- SpamShield has149,359 multilingual SMS rows and gated access; it is not a
  million English emails.
- Enron-Spam cleaned mirrors and the42,619-row mixed SMS/conversation/email
  all-scam-spam collection do not establish a new million distinct messages.
- Synthetic, translated or chat/SMS data were not accepted as native email.

## Data gate

Before training, parse messages, verify labels, remove canonical copies and
cross-source/old-data overlap, and freeze provenance, label map, English filter
and quote/thread-separated TRAIN/SELECT/TEST. Only TRAIN errors may be mined;
retain correct support and per-row lifetime exposure limits. Unknown labels
remain unknown. Current `training_ready=false` for this million-scale expansion.
No unlabelled Enron message was invented as HAM.

Evidence: `research/english-million/source-audit-20261116.json` and the pinned
171-file PST catalogue. Only metadata/hashes/aggregates are committed; raw mail
stays local or at its original public source. Existing v82/v84, RU/ES/TR and the
Korean probe were not modified; no new controller or cron was created.
