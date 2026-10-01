#!/usr/bin/env python3
import hashlib, json, math, random, re, statistics, subprocess, tarfile, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from email import policy
from email.parser import BytesParser
from pathlib import Path
from urllib.request import Request, urlopen

BASE='https://spamassassin.apache.org/old/publiccorpus/'
CACHE=Path('.cache/spamassassin-four-way')
REPORTS=Path('reports')
RSPAMD='http://127.0.0.1:11333/'
CTRL='http://127.0.0.1:11334/'
DIM=1<<16
SEED=1337
MAX_FPR=0.005
SPAM_ACTIONS={'reject','add header','rewrite subject','quarantine','discard'}
TOKEN_RE=re.compile(r"[\w@.\-]{2,48}", re.UNICODE)
URL_RE=re.compile(r"https?://([^/\s:]+)", re.I)
ARCHIVES={
 'train_ham':'20030228_easy_ham.tar.bz2',
 'train_spam':'20030228_spam.tar.bz2',
 'test_ham1':'20030228_easy_ham_2.tar.bz2',
 'test_ham2':'20030228_hard_ham.tar.bz2',
 'test_spam':'20050311_spam_2.tar.bz2',
}

def get(url, timeout=90):
    req=Request(url, headers={'User-Agent':'MailGuard-four-way/1.0'})
    return urlopen(req, timeout=timeout).read()

def post(url, data, timeout=30):
    req=Request(url, data=data, method='POST',
                headers={'Content-Type':'message/rfc822','User-Agent':'MailGuard-four-way/1.0'})
    return urlopen(req, timeout=timeout).read()

def wait_rspamd():
    for _ in range(90):
        try:
            get(RSPAMD+'ping',5)
            return
        except Exception:
            time.sleep(2)
    raise RuntimeError('Rspamd did not become ready')

def reset_bayes():
    subprocess.run(['docker','compose','exec','-T','redis','redis-cli','FLUSHALL'],
                   check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    subprocess.run(['docker','compose','restart','rspamd'],
                   check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    wait_rspamd()

def prepare():
    CACHE.mkdir(parents=True,exist_ok=True)
    groups={}
    for key,name in ARCHIVES.items():
        arc=CACHE/name
        if not arc.exists():
            print('download',name,flush=True)
            arc.write_bytes(get(BASE+name,120))
        dest=CACHE/(name+'.dir')
        if not dest.exists():
            dest.mkdir(parents=True)
            with tarfile.open(arc,'r:bz2') as tf:
                tf.extractall(dest)
        files=[p for p in dest.rglob('*')
               if p.is_file() and p.name!='cmds' and not p.name.startswith('.')]
        files.sort()
        groups[key]=files
        print(key,len(files),flush=True)
    return groups

def split_train(groups):
    rnd=random.Random(SEED)
    ham=list(groups['train_ham']); spam=list(groups['train_spam'])
    rnd.shuffle(ham); rnd.shuffle(spam)
    ham_val=max(1,len(ham)//5); spam_val=max(1,len(spam)//5)
    train=[{'path':p,'y':0} for p in ham[ham_val:]]
    train += [{'path':p,'y':1} for p in spam[spam_val:]]
    val=[{'path':p,'y':0} for p in ham[:ham_val]]
    val += [{'path':p,'y':1} for p in spam[:spam_val]]
    test=[{'path':p,'y':0} for p in groups['test_ham1']+groups['test_ham2']]
    test += [{'path':p,'y':1} for p in groups['test_spam']]
    return train,val,test

def learn(items):
    spam=[x['path'] for x in items if x['y']]
    ham=[x['path'] for x in items if not x['y']]
    for label,files,endpoint in [('ham',ham,'learnham'),('spam',spam,'learnspam')]:
        for i,p in enumerate(files,1):
            post(CTRL+endpoint,p.read_bytes(),30)
            if i%250==0 or i==len(files):
                print('learn',label,i,'/',len(files),flush=True)

def scan_one(item):
    raw=item['path'].read_bytes()
    j=json.loads(post(RSPAMD+'checkv2',raw,30))
    syms=[]
    for name,v in (j.get('symbols') or {}).items():
        if isinstance(v,dict):
            syms.append((name,float(v.get('score',0) or 0)))
    action=str(j.get('action','')).lower()
    return {**item,'raw':raw,'action':action,
            'rscore':float(j.get('score',0) or 0),
            'req':float(j.get('required_score',0) or 0),
            'symbols':syms,'rspam':action in SPAM_ACTIONS}

def scan_many(items, label, workers=8):
    print('scan phase',label,'count',len(items),flush=True)
    out=[None]*len(items)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs={ex.submit(scan_one,it):i for i,it in enumerate(items)}
        done=0
        for fut in as_completed(futs):
            out[futs[fut]]=fut.result()
            done+=1
            if done%250==0 or done==len(items):
                print(label,done,'/',len(items),flush=True)
    return out

def parse_mail(raw):
    try:
        msg=BytesParser(policy=policy.default).parsebytes(raw)
    except Exception:
        return '', '', raw.decode('utf-8','ignore')
    subject=str(msg.get('subject','')); sender=str(msg.get('from','')); bodies=[]
    try:
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_maintype()=='text':
                    try: bodies.append(part.get_content())
                    except Exception: pass
        else:
            try: bodies.append(msg.get_content())
            except Exception: bodies.append(raw.decode('utf-8','ignore'))
    except Exception:
        bodies.append(raw.decode('utf-8','ignore'))
    return subject,sender,'\n'.join(str(x) for x in bodies)

def h(s):
    return int.from_bytes(hashlib.blake2b(s.encode('utf-8','ignore'),digest_size=8).digest(),'little') & (DIM-1)

def make_features(r):
    subject,sender,text=parse_mail(r['raw']); x={}
    def add(name,val=1.0):
        k=h(name); x[k]=x.get(k,0.0)+val
    toks=TOKEN_RE.findall((subject+' '+text).lower())[:5000]
    for t in toks: add('t:'+t)
    for a,b in zip(toks[:1500],toks[1:1501]): add('b:'+a+'_'+b)
    m=re.search(r'@([\w.-]+)',sender.lower())
    if m: add('from:'+m.group(1))
    for m in URL_RE.finditer(text[:200000]): add('url:'+m.group(1).lower())
    add('rspamd_score',max(-3,min(3,r['rscore']/15.0)))
    if r['req']: add('rspamd_ratio',max(-3,min(3,r['rscore']/r['req'])))
    add('act:'+r['action'])
    for n,s in r['symbols'][:250]: add('sym:'+n.lower(),max(-10,min(10,s)))
    return x

def sigmoid(z):
    if z>=0:
        return 1/(1+math.exp(-min(z,60)))
    e=math.exp(max(z,-60))
    return e/(1+e)

def predict(w,b,x):
    return sigmoid(b+sum(w.get(k,0.0)*v for k,v in x.items()))

def train_model(rows,epochs=10,lr0=.05,l2=1e-6):
    w={}; b=0.0; rnd=random.Random(SEED); idx=list(range(len(rows)))
    nspam=sum(r['y'] for r in rows); nham=len(rows)-nspam
    sw=len(rows)/(2*max(1,nspam)); hw=len(rows)/(2*max(1,nham))
    for ep in range(epochs):
        rnd.shuffle(idx); lr=lr0/math.sqrt(1+ep*.35); loss=0.0
        for ii in idx:
            r=rows[ii]; y=r['y']; p=predict(w,b,r['x']); cw=sw if y else hw
            if y and not r['rspam']: cw*=2.5
            err=(p-y)*cw
            for k,v in r['x'].items():
                old=w.get(k,0.0); w[k]=old-lr*(err*v+l2*old)
            b-=lr*err
            pp=min(1-1e-9,max(1e-9,p))
            loss+=-cw*(y*math.log(pp)+(1-y)*math.log(1-pp))
        print('epoch',ep+1,'loss',round(loss/len(rows),5),flush=True)
    return w,b

def pick_threshold(val,w,b):
    vals=[(r['y'],predict(w,b,r['x']),r['rspam']) for r in val]
    ths=sorted(set(p for _,p,_ in vals),reverse=True)+[1.000001]
    best=None
    for t in ths:
        tp=fp=ps=ph=0
        for y,p,rs in vals:
            pred=rs or p>=t
            if y: ps+=1; tp+=int(pred)
            else: ph+=1; fp+=int(pred)
        rec=tp/max(1,ps); fpr=fp/max(1,ph)
        cand=(t,rec,fpr,tp,fp)
        if fpr<=MAX_FPR and (best is None or rec>best[1] or (rec==best[1] and fpr<best[2])):
            best=cand
    return best or (1.000001,0,0,0,0)

def baseline(rows):
    spam=sum(r['y'] for r in rows); ham=len(rows)-spam
    tp=sum(1 for r in rows if r['y'] and r['rspam'])
    fp=sum(1 for r in rows if not r['y'] and r['rspam'])
    return {'spamTotal':spam,'hamTotal':ham,'spamDetected':tp,'falsePositives':fp,
            'recall':tp/max(1,spam),'fpr':fp/max(1,ham)}

def hybrid(rows,w,b,t):
    spam=sum(r['y'] for r in rows); ham=len(rows)-spam
    tp=fp=0; preds=[]
    for r in rows:
        p=predict(w,b,r['x']); pred=r['rspam'] or p>=t
        if r['y'] and pred: tp+=1
        if not r['y'] and pred: fp+=1
        preds.append({'y':r['y'],'p':p,'rspam':r['rspam'],'pred':pred})
    return {'spamTotal':spam,'hamTotal':ham,'spamDetected':tp,'falsePositives':fp,
            'recall':tp/max(1,spam),'fpr':fp/max(1,ham),'threshold':t},preds

def review(preds):
    n=len(preds); k=max(1,math.ceil(n*.01)); rnd=random.Random(SEED)
    top=sum(r['y'] for r in sorted(preds,key=lambda z:(z['rspam'],z['p']),reverse=True)[:k])
    counts=[sum(r['y'] for r in rnd.sample(preds,k)) for _ in range(2000)]
    mean=statistics.mean(counts)
    return {'budget':k,'topRiskSpamFound':top,'randomSpamFoundMean':mean,
            'lift':top/mean if mean else None}

def enrich(rows):
    for r in rows:
        r['x']=make_features(r)
    return rows

def main():
    REPORTS.mkdir(exist_ok=True)
    wait_rspamd()
    groups=prepare()
    train,val,test=split_train(groups)
    print('TRAIN',len(train),'VAL',len(val),'TEST',len(test),flush=True)

    print('\n=== PHASE A: plain Rspamd, no Bayes training ===',flush=True)
    reset_bayes()
    plain_train=enrich(scan_many(train,'plain-train'))
    plain_val=enrich(scan_many(val,'plain-val'))
    plain_test=enrich(scan_many(test,'plain-test'))
    plain_rspamd=baseline(plain_test)
    w0,b0=train_model(plain_train)
    th0=pick_threshold(plain_val,w0,b0)
    print('plain validation threshold',th0,flush=True)
    plain_mg,plain_preds=hybrid(plain_test,w0,b0,th0[0])

    print('\n=== PHASE B: Rspamd with Bayes trained on TRAIN only ===',flush=True)
    reset_bayes()
    learn(train)
    bayes_train=enrich(scan_many(train,'bayes-train'))
    bayes_val=enrich(scan_many(val,'bayes-val'))
    bayes_test=enrich(scan_many(test,'bayes-test'))
    bayes_rspamd=baseline(bayes_test)
    w1,b1=train_model(bayes_train)
    th1=pick_threshold(bayes_val,w1,b1)
    print('bayes validation threshold',th1,flush=True)
    bayes_mg,bayes_preds=hybrid(bayes_test,w1,b1,th1[0])

    result={
      'dataset':{'train':len(train),'validation':len(val),'test':len(test),
                 'testSpam':plain_rspamd['spamTotal'],'testHam':plain_rspamd['hamTotal']},
      'plainRspamd':plain_rspamd,
      'plainRspamdPlusMailGuard':plain_mg,
      'rspamdBayes':bayes_rspamd,
      'rspamdBayesPlusMailGuard':bayes_mg,
      'plainValidation':{'threshold':th0[0],'recall':th0[1],'fpr':th0[2]},
      'bayesValidation':{'threshold':th1[0],'recall':th1[1],'fpr':th1[2]},
      'review1pctPlain':review(plain_preds),
      'review1pctBayes':review(bayes_preds)
    }
    (REPORTS/'four-way-benchmark.json').write_text(json.dumps(result,indent=2))

    rows=[
      ('Rspamd',plain_rspamd),
      ('Rspamd + MailGuard',plain_mg),
      ('Rspamd + Bayes',bayes_rspamd),
      ('Rspamd + Bayes + MailGuard',bayes_mg),
    ]
    md=['# Four-way MailGuard benchmark','',
        f"Test set: {plain_rspamd['spamTotal']} spam + {plain_rspamd['hamTotal']} ham.",'',
        '| Mode | Spam detected | Spam recall | False positives | FP rate |',
        '|---|---:|---:|---:|---:|']
    for name,r in rows:
        md.append(f"| {name} | {r['spamDetected']}/{r['spamTotal']} | {r['recall']:.2%} | {r['falsePositives']}/{r['hamTotal']} | {r['fpr']:.3%} |")
    md += ['',f"Plain MailGuard threshold: {plain_mg['threshold']:.6f}",
           f"Bayes MailGuard threshold: {bayes_mg['threshold']:.6f}",'',
           '## 1% review',
           f"Plain flow: random mean {result['review1pctPlain']['randomSpamFoundMean']:.2f}, top-risk {result['review1pctPlain']['topRiskSpamFound']}, lift {result['review1pctPlain']['lift']}.",
           f"Bayes flow: random mean {result['review1pctBayes']['randomSpamFoundMean']:.2f}, top-risk {result['review1pctBayes']['topRiskSpamFound']}, lift {result['review1pctBayes']['lift']}."]
    text='\n'.join(md)+'\n'
    (REPORTS/'four-way-benchmark.md').write_text(text)
    print(text,flush=True)

if __name__=='__main__':
    main()
