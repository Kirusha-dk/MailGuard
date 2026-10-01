#!/usr/bin/env python3
import csv, hashlib, json, math, random, re, statistics, tarfile, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from email import policy
from email.parser import BytesParser
from pathlib import Path
from urllib.request import Request, urlopen

BASE='https://spamassassin.apache.org/old/publiccorpus/'
ROOT=Path('.')
CACHE=ROOT/'.cache'/'spamassassin-fast'
DATA=ROOT/'data'/'fastbench'
REPORTS=ROOT/'reports'
RSPAMD='http://127.0.0.1:11333/'
CTRL='http://127.0.0.1:11334/'
DIM=1<<16
SEED=1337
MAX_FPR=0.005
ARCHIVES={
 'train_ham':'20030228_easy_ham.tar.bz2',
 'train_spam':'20030228_spam.tar.bz2',
 'test_ham1':'20030228_easy_ham_2.tar.bz2',
 'test_ham2':'20030228_hard_ham.tar.bz2',
 'test_spam':'20050311_spam_2.tar.bz2',
}
TOKEN_RE=re.compile(r"[\w@.\-]{2,48}", re.UNICODE)
URL_RE=re.compile(r"https?://([^/\s:]+)", re.I)
SPAM_ACTIONS={'reject','add header','rewrite subject','quarantine','discard'}

def get(url, timeout=60):
    req=Request(url, headers={'User-Agent':'MailGuard-benchmark/1.0'})
    return urlopen(req, timeout=timeout).read()

def post(url, data, ctype='message/rfc822', timeout=30):
    req=Request(url, data=data, method='POST', headers={'Content-Type':ctype,'User-Agent':'MailGuard-benchmark/1.0'})
    return urlopen(req, timeout=timeout).read()

def wait_rspamd():
    for _ in range(90):
        try:
            if get(RSPAMD+'ping',5): return
        except Exception: pass
        time.sleep(2)
    raise RuntimeError('Rspamd did not become ready')

def prepare():
    CACHE.mkdir(parents=True,exist_ok=True); DATA.mkdir(parents=True,exist_ok=True)
    groups={}
    for key,name in ARCHIVES.items():
        arc=CACHE/name
        if not arc.exists():
            print('download',name,flush=True); arc.write_bytes(get(BASE+name,120))
        dest=CACHE/(name+'.dir')
        if not dest.exists():
            dest.mkdir(parents=True); tarfile.open(arc,'r:bz2').extractall(dest)
        files=[p for p in dest.rglob('*') if p.is_file() and p.name!='cmds' and not p.name.startswith('.')]
        files.sort()
        groups[key]=files
        print(key,len(files),flush=True)
    return groups

def learn(files, spam):
    ep=CTRL+('learnspam' if spam else 'learnham')
    ok=0
    for i,p in enumerate(files,1):
        try:
            post(ep,p.read_bytes(),timeout=30); ok+=1
        except Exception as e:
            print('learn error',p,e,flush=True)
        if i%250==0 or i==len(files): print('learn', 'spam' if spam else 'ham',i,'/',len(files),flush=True)
    return ok

def scan_one(p):
    raw=p.read_bytes(); out=post(RSPAMD+'checkv2',raw,timeout=30); j=json.loads(out)
    syms=j.get('symbols') or {}
    symbols=[]
    for name,v in syms.items():
        if isinstance(v,dict): symbols.append((name,float(v.get('score',0) or 0)))
    action=str(j.get('action','')).lower(); score=float(j.get('score',0) or 0); req=float(j.get('required_score',0) or 0)
    return raw,action,score,req,symbols

def parse_mail(raw):
    try: msg=BytesParser(policy=policy.default).parsebytes(raw)
    except Exception: return '', '', raw.decode('utf-8','ignore')
    subject=str(msg.get('subject','')); sender=str(msg.get('from',''))
    bodies=[]
    try:
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_maintype()=='text':
                    try: bodies.append(part.get_content())
                    except Exception: pass
        else:
            try: bodies.append(msg.get_content())
            except Exception: bodies.append(raw.decode('utf-8','ignore'))
    except Exception: bodies.append(raw.decode('utf-8','ignore'))
    return subject,sender,'\n'.join(str(x) for x in bodies)

def h(s):
    return int.from_bytes(hashlib.blake2b(s.encode('utf-8','ignore'),digest_size=8).digest(),'little') & (DIM-1)

def features(raw, action, score, req, symbols):
    subject,sender,text=parse_mail(raw); x={}
    def add(name,val=1.0):
        k=h(name); x[k]=x.get(k,0.0)+val
    toks=TOKEN_RE.findall((subject+' '+text).lower())[:5000]
    for t in toks: add('t:'+t)
    for a,b in zip(toks[:1500],toks[1:1501]): add('b:'+a+'_'+b)
    m=re.search(r'@([\w.-]+)',sender.lower())
    if m: add('from:'+m.group(1))
    for m in URL_RE.finditer(text[:200000]): add('url:'+m.group(1).lower())
    add('rspamd_score',max(-3,min(3,score/15.0)))
    if req: add('rspamd_ratio',max(-3,min(3,score/req)))
    add('act:'+action)
    for n,s in symbols[:250]: add('sym:'+n.lower(),max(-10,min(10,s)))
    return x

def scan_many(items,workers=8):
    out=[None]*len(items)
    def work(i,item):
        raw,a,s,r,sy=scan_one(item['path']); return i,{**item,'raw':raw,'action':a,'rscore':s,'req':r,'symbols':sy,'rspam':a in SPAM_ACTIONS}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs=[ex.submit(work,i,it) for i,it in enumerate(items)]
        done=0
        for f in as_completed(futs):
            i,v=f.result(); out[i]=v; done+=1
            if done%250==0 or done==len(items): print('scan',done,'/',len(items),flush=True)
    return out

def sigmoid(z):
    if z>=0: return 1/(1+math.exp(-min(z,60)))
    e=math.exp(max(z,-60)); return e/(1+e)

def predict(w,b,x): return sigmoid(b+sum(w.get(k,0.0)*v for k,v in x.items()))

def train(rows,epochs=10,lr0=.05,l2=1e-6):
    w={}; b=0.0; rnd=random.Random(SEED)
    nspam=sum(r['y'] for r in rows); nham=len(rows)-nspam
    sw=len(rows)/(2*max(1,nspam)); hw=len(rows)/(2*max(1,nham))
    idx=list(range(len(rows)))
    for ep in range(epochs):
        rnd.shuffle(idx); lr=lr0/math.sqrt(1+ep*.35); loss=0
        for ii in idx:
            r=rows[ii]; y=r['y']; p=predict(w,b,r['x']); cw=sw if y else hw
            if y and not r['rspam']: cw*=2.5
            err=(p-y)*cw
            for k,v in r['x'].items():
                old=w.get(k,0.0); w[k]=old-lr*(err*v+l2*old)
            b-=lr*err
            pp=min(1-1e-9,max(1e-9,p)); loss+=-cw*(y*math.log(pp)+(1-y)*math.log(1-pp))
        print('epoch',ep+1,'loss',round(loss/len(rows),5),flush=True)
    return w,b

def threshold(validation,w,b):
    vals=[]
    for r in validation:
        p=predict(w,b,r['x']); vals.append((r['y'],p,r['rspam']))
    ths=sorted(set(p for _,p,_ in vals),reverse=True)+[1.000001]
    best=None
    for t in ths:
        tp=fp=ps=ph=0
        for y,p,rs in vals:
            pred=rs or p>=t
            if y: ps+=1; tp+=int(pred)
            else: ph+=1; fp+=int(pred)
        rec=tp/max(1,ps); fpr=fp/max(1,ph)
        if fpr<=MAX_FPR and (best is None or rec>best[1] or (rec==best[1] and fpr<best[2])): best=(t,rec,fpr,tp,fp)
    return best or (1.000001,0,0,0,0)

def evaluate(test,w,b,t):
    rs_tp=hy_tp=rs_fp=hy_fp=spam=ham=0; preds=[]
    for r in test:
        y=r['y']; p=predict(w,b,r['x']); hy=r['rspam'] or p>=t
        if y:
            spam+=1; rs_tp+=int(r['rspam']); hy_tp+=int(hy)
        else:
            ham+=1; rs_fp+=int(r['rspam']); hy_fp+=int(hy)
        preds.append({'label':'spam' if y else 'ham','p':p,'rspamdSpam':r['rspam'],'hybridSpam':hy,'path':str(r['path'])})
    return {'spamTotal':spam,'hamTotal':ham,'rspamdSpamDetected':rs_tp,'hybridSpamDetected':hy_tp,'rspamdFalsePositives':rs_fp,'hybridFalsePositives':hy_fp,'rspamdRecall':rs_tp/max(1,spam),'hybridRecall':hy_tp/max(1,spam),'rspamdFpr':rs_fp/max(1,ham),'hybridFpr':hy_fp/max(1,ham),'threshold':t},preds

def review(preds,percent=1.0,runs=2000):
    n=len(preds); k=max(1,math.ceil(n*percent/100)); top=sorted(preds,key=lambda r:(r['rspamdSpam'],r['p']),reverse=True)[:k]
    top_spam=sum(r['label']=='spam' for r in top); rnd=random.Random(SEED); counts=[]
    for _ in range(runs): counts.append(sum(r['label']=='spam' for r in rnd.sample(preds,k)))
    mean=statistics.mean(counts)
    residual=[r for r in preds if not r['rspamdSpam']]; top2=sorted(residual,key=lambda r:r['p'],reverse=True)[:min(k,len(residual))]
    return {'budget':k,'topRiskSpamFound':top_spam,'randomSpamFoundMean':mean,'lift':top_spam/mean if mean else None,'residualTopRiskSpamFound':sum(r['label']=='spam' for r in top2),'residualCandidates':len(residual)}

def main():
    random.seed(SEED); REPORTS.mkdir(exist_ok=True); wait_rspamd(); g=prepare()
    print('training rspamd bayes',flush=True); learn(g['train_ham'],False); learn(g['train_spam'],True)
    train_items=[{'path':p,'y':0} for p in g['train_ham']]+[{'path':p,'y':1} for p in g['train_spam']]
    test_items=[{'path':p,'y':0} for p in g['test_ham1']+g['test_ham2']]+[{'path':p,'y':1} for p in g['test_spam']]
    rnd=random.Random(SEED); ham=[x for x in train_items if not x['y']]; spam=[x for x in train_items if x['y']]; rnd.shuffle(ham); rnd.shuffle(spam)
    val=ham[:max(1,len(ham)//5)]+spam[:max(1,len(spam)//5)]; tr=ham[len(ham)//5:]+spam[len(spam)//5:]
    all_train=scan_many(tr+val); tr=all_train[:len(tr)]; val=all_train[len(tr):]
    for r in tr+val: r['x']=features(r['raw'],r['action'],r['rscore'],r['req'],r['symbols'])
    w,b=train(tr); th=threshold(val,w,b); print('validation threshold',th,flush=True)
    test=scan_many(test_items)
    for r in test: r['x']=features(r['raw'],r['action'],r['rscore'],r['req'],r['symbols'])
    result,preds=evaluate(test,w,b,th[0]); rev=review(preds); result['validation']={'recall':th[1],'fpr':th[2]}; result['review1pct']=rev
    (REPORTS/'fast-benchmark.json').write_text(json.dumps(result,indent=2))
    with (REPORTS/'fast-predictions.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=preds[0].keys()); wr.writeheader(); wr.writerows(preds)
    md=f'''# MailGuard real benchmark

| Metric | Rspamd Bayes | Rspamd Bayes + MailGuard |
|---|---:|---:|
| Spam detected | {result['rspamdSpamDetected']}/{result['spamTotal']} | {result['hybridSpamDetected']}/{result['spamTotal']} |
| Spam recall | {result['rspamdRecall']:.2%} | {result['hybridRecall']:.2%} |
| False positives | {result['rspamdFalsePositives']}/{result['hamTotal']} | {result['hybridFalsePositives']}/{result['hamTotal']} |
| False-positive rate | {result['rspamdFpr']:.3%} | {result['hybridFpr']:.3%} |

Threshold: {result['threshold']:.6f}

1% review: random mean {rev['randomSpamFoundMean']:.2f} spam, top-risk {rev['topRiskSpamFound']} spam, lift {rev['lift']}.
'''
    (REPORTS/'fast-benchmark.md').write_text(md)
    print(md,flush=True)

if __name__=='__main__':
    main()
