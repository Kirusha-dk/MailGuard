"""Pinned public Spanish and Turkish email sources; content allowlist only."""
import csv, hashlib, io, urllib.request, zipfile
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

TURKISH_URL = 'https://archive.ics.uci.edu/static/public/530/turkish+spam+v01.zip'
TURKISH_ZIP_SHA = '3d3c7e4254e607059a901b47bf9cac1a06429bb0b7bbfad5e22e80f84c0b6e29'
TURKISH_CSV_SHA = '2ddc3bcdc69a087432b2e8a66a338721ec7114f6eb3ba6ade013ca0165505ab3'


def parse_turkish_csv(data):
    records = csv.reader(io.StringIO(data.decode('utf-8-sig')))
    if next(records) != ['Text', 'Classification']:
        raise ValueError('Turkish source schema drift')
    rows = []; stats = dict(rejoined_extra_fields=0, excluded_footer=0)
    for record in records:
        # The pinned export ends with an unlabelled spreadsheet count, not mail.
        if record == ['', '', '496']:
            stats['excluded_footer'] += 1
            continue
        if len(record) < 2 or record[-1] not in ('spam', 'ham'):
            raise ValueError('Unknown original Turkish label; never infer it')
        # Badly escaped commas occur inside the text column. The final field
        # consistently contains the original label in the pinned archive.
        stats['rejoined_extra_fields'] += int(len(record) > 2)
        rows.append(dict(subject='',body=','.join(record[:-1]),
            y={'spam':1,'ham':0}[record[-1]],source='UCI-TurkishSpam-V01',kind='email',language='tr'))
    return rows, stats


def fetch_turkish_uci(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    path=root/'turkish-spam-v01.zip'
    if not path.exists():
        with urllib.request.urlopen(TURKISH_URL,timeout=90) as response:
            data=response.read(10_000_001)
        if len(data)>10_000_000 or hashlib.sha256(data).hexdigest()!=TURKISH_ZIP_SHA:
            raise ValueError('Turkish archive checksum mismatch')
        path.write_bytes(data)
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=TURKISH_ZIP_SHA:
        raise ValueError('Turkish cached archive drift')
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        csv_data=archive.read('trspam.csv')
    if hashlib.sha256(csv_data).hexdigest()!=TURKISH_CSV_SHA:
        raise ValueError('Turkish CSV checksum mismatch')
    rows, parsing=parse_turkish_csv(csv_data)
    if (len(rows)!=825 or sum(row['y'] for row in rows)!=329 or
            parsing!=dict(rejoined_extra_fields=124,excluded_footer=1)):
        raise ValueError('Pinned Turkish labels or parsing counts drift')
    return rows,dict(name='UCI Turkish Spam V01',version=1,url=TURKISH_URL,
        sha256=TURKISH_ZIP_SHA,csv_sha256=TURKISH_CSV_SHA,license='CC-BY-4.0',
        attribution='Turkish Spam V01 (2019), UCI Machine Learning Repository; https://doi.org/10.24432/C5WG7F',
        positive_class='spam',negative_class='legitimate_email',feature_allowlist=['Text'],
        original_label_map={'spam':1,'ham':0},labelled_rows=825,original_spam=329,original_ham=496,
        publisher_description_counts={'spam':330,'ham':496},parsing=parsing,
        count_discrepancy='Pinned export has329 labelled spam,496 labelled HAM, and one unlabelled footer; no missing label was invented')

TURKISH_MAIL_URL = ('https://www.kaggle.com/api/v1/datasets/download/emrahaydemr/'
                    'turkish-mail-dataset-normalspam?datasetVersionNumber=1')
TURKISH_MAIL_SHA = 'dc6d8f2e4b7ae8ace77b54521760f10940fd677238a39fb9f33578259c85a250'


def fetch_turkish(root):
    import re
    rows, uci = fetch_turkish_uci(root)
    path=Path(root)/'turkish-mail-normalspam-v1.zip'
    if not path.exists():
        with urllib.request.urlopen(TURKISH_MAIL_URL,timeout=90) as response:
            data=response.read(10_000_001)
        if len(data)>10_000_000 or hashlib.sha256(data).hexdigest()!=TURKISH_MAIL_SHA:
            raise ValueError('Turkish additional archive checksum mismatch')
        path.write_bytes(data)
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=TURKISH_MAIL_SHA:
        raise ValueError('Turkish additional cached archive drift')
    additional=[]
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for name in sorted(archive.namelist()):
            match=re.fullmatch(r'Mails/(norm|spam)[0-9]+\.txt',name)
            if match:
                text=archive.read(name).decode('utf-8-sig')
                additional.append(dict(subject='',body=text,y=int(match[1]=='spam'),
                    source='Simsek-Aydemir-TurkishMail-v1',kind='email',language='tr'))
            elif name!='norm_spam.arff':
                raise ValueError('Unexpected additional Turkish source member')
    if len(additional)!=1017 or sum(row['y'] for row in additional)!=515:
        raise ValueError('Additional original Turkish counts drift')
    extra=dict(name='Turkish Mail Dataset Normal/Spam',version=1,url=TURKISH_MAIL_URL,
        sha256=TURKISH_MAIL_SHA,license='Unspecified standard licence; publisher requires article citation for use',
        attribution='Simsek, H. and Aydemir, E. (2022), Classification of Unwanted E-Mails (Spam) with Turkish Text by Different Algorithms in Weka Program; https://doi.org/10.55195/jscai.1104694',
        provenance='Original publisher: mail sent to and from20 accounts; native Turkish titles/content exported as text',
        original_label_map={'spam*.txt':1,'norm*.txt':0},original_spam=515,original_ham=502,
        excluded_arff_copy=True,raw_mail_redistribution=False)
    return rows+additional,dict(name='Two public native Turkish email sources',
        sources=[uci,extra],feature_allowlist=['email_text'],
        normalizer_deduplicates_across_sources=True,synthetic_or_translated=False,
        source_rights_note='UCI CC-BY4.0; additional publisher requests citation but gives no standard licence. Research pilot only; source terms must accompany any final package.')
