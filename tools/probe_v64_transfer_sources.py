"""Download and inventory new public sources without loading a classifier."""
import hashlib
import json
import re
import tarfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request,urlopen

OUT=Path('.cache/v64-transfer-sources')
REV='84209612df4c55831f074a65b7c137a4b980059d'

def download(url,name,timeout=35):
    dest=OUT/name
    h=hashlib.sha256()
    with urlopen(Request(url,headers={'User-Agent':'MailGuard-public-research/1.0'}),timeout=timeout) as src,dest.open('wb') as f:
        for chunk in iter(lambda:src.read(1024*1024),b''):
            f.write(chunk);h.update(chunk)
    return dict(url=url,file=name,sha256=h.hexdigest(),bytes=dest.stat().st_size)

def fetch(name,urls,kind):
    errors=[]
    for url in urls:
        try:
            evidence=download(url,name)
            if kind=='zip':
                with zipfile.ZipFile(OUT/name) as z:
                    names=z.namelist()
                    ham=sum('/dataset/1/' in p and not p.endswith('/') for p in names)
                    spam=sum('/dataset/2/' in p and not p.endswith('/') for p in names)
                    if (ham,spam)!=(10776,8752):raise ValueError('Unexpected published label counts')
                    evidence.update(ham=ham,spam=spam,revision=REV)
            elif kind=='tar':
                with tarfile.open(OUT/name,'r:*') as t:
                    evidence['members']=[m.name for m in t if m.isfile()][:16]
            elif kind=='mbox':
                with (OUT/name).open('rb') as f:
                    count=sum(line.startswith(b'From ') for line in f)
                if count<10:raise ValueError('Expected a real mbox')
                evidence.update(mboxMessages=count,labels='Author hand-classified phishing; original labels retained',attribution='Jose Nazario',license='CC-BY-4.0')
            return dict(name=name,available=True,**evidence)
        except (OSError,ValueError,zipfile.BadZipFile,tarfile.TarError) as error:
            (OUT/name).unlink(missing_ok=True)
            errors.append(dict(url=url,error=type(error).__name__))
    return dict(name=name,available=False,attempts=errors)

def genspam():
    urls=[]
    for page in ('https://www.benmedlock.co.uk/genspam.html','http://www.benmedlock.co.uk/genspam.html'):
        try:
            with urlopen(Request(page,headers={'User-Agent':'MailGuard-public-research/1.0'}),timeout=20) as f:
                text=f.read().decode('utf-8','replace')
            urls.extend(urljoin(page,p) for p in re.findall(r'href=["\']([^"\']+)["\']',text,re.I)
                        if re.search(r'genspam.*\.(?:tar\.gz|tgz)$',p,re.I))
        except OSError:pass
    # Archive name is documented by the Cambridge L101 corpus reader.
    urls.extend(('https://www.benmedlock.co.uk/GenSpam.tar.gz','http://www.benmedlock.co.uk/GenSpam.tar.gz'))
    return fetch('genspam.tar.gz',list(dict.fromkeys(urls)),'tar')

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    tasks=[lambda:fetch('labelled-mail.zip',[f'https://codeload.github.com/realprogrammersusevim/email-dataset/zip/{REV}'],'zip'),
           lambda:fetch('nazario-2025.mbox',['https://monkey.org/~jose/phishing/phishing-2025'],'mbox'),
           lambda:fetch('lingspam.tar.gz',['https://www2.aueb.gr/users/ion/data/lingspam_public.tar.gz','http://www.aueb.gr/users/ion/data/lingspam_public.tar.gz'],'tar'),
           genspam,
           lambda:fetch('trec06p.tgz',['https://plg.uwaterloo.ca/~gvcormac/treccorpus06/trec06p.tgz','https://cormack.uwaterloo.ca/treccorpus06/trec06p.tgz'],'tar')]
    with ThreadPoolExecutor(max_workers=5) as workers:
        futures=[workers.submit(task) for task in tasks]
        result=[]
        for f in futures:
            r=f.result();result.append(r);print(json.dumps(r),flush=True)
    (OUT/'source-inventory.json').write_text(json.dumps(dict(classifierLoaded=False,sources=result),indent=2))
    if not result[0]['available'] or not result[1]['available']:raise ValueError('Required new source unavailable')

if __name__=='__main__':main()
