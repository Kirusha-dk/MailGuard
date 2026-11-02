"""Pinned public Spanish email source, content allowlist only."""
import csv, hashlib, io, urllib.request
from pathlib import Path
URL='https://data.mendeley.com/public-files/datasets/hz2d6gz7pc/files/f796c8e2-3768-4c2d-8b73-48f0d7771de5/file_downloaded'
SHA='656b2245d58da72d640680e5c2a168673a130b38607f2a427c773bbb167e995e'
def fetch_spanish(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    path=root/'spaphish-v5.csv'
    if not path.exists():
        req=urllib.request.Request(URL,headers={'User-Agent':'MailGuard-research/83'})
        with urllib.request.urlopen(req,timeout=90) as response: data=response.read(10_000_001)
        if len(data)>10_000_000 or hashlib.sha256(data).hexdigest()!=SHA:raise ValueError('SpaPhish checksum mismatch')
        path.write_bytes(data)
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=SHA:raise ValueError('SpaPhish cache drift')
    rows=[]
    for record in csv.DictReader(io.StringIO(data.decode('utf-8-sig'))):
        if record['Label'] not in ('0','1'):raise ValueError('Unknown published label')
        rows.append(dict(subject=record['subject'],body=record['body'],y=int(record['Label']),source='SpaPhish-v5',kind='email',language='es'))
    if len(rows)!=1395 or sum(r['y'] for r in rows)!=731:raise ValueError('Published counts drift')
    return rows,dict(name='SpaPhish',version=5,url=URL,sha256=SHA,license='CC-BY-4.0',attribution='SpaPhish authors; https://data.mendeley.com/datasets/hz2d6gz7pc/5',positive_class='phishing',negative_class='legitimate_email',feature_allowlist=['subject','body'],expert_annotations_used_as_features=False)
