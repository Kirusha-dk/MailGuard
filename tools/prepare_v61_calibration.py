"""Independent, deterministic CAL embedding job; no classifier fit or TEST access."""
import hashlib
import json
from pathlib import Path
import numpy as np
from train_v55_multilingual import corpus_rows,file_sha
from v61_semantic import fetch_encoder,Encoder,VIEW_VERSION,REVISION


def main():
    source=Path('.cache/v60-data'); out=Path('.cache/v61-cal-query'); out.mkdir(parents=True,exist_ok=False)
    provenance=json.loads((source/'provenance.json').read_text())
    if file_sha(source/'cal.jsonl.gz')!=provenance['splitHashes']['cal']:raise ValueError('CAL changed')
    rows=list(corpus_rows(source/'cal.jsonl.gz'))
    if any(r.get('weakLabel') for r in rows):raise ValueError('Weak HAM in CAL')
    fetch_encoder(out/'encoder'); encoder=Encoder(out/'encoder')
    np.save(out/'embeddings.npy',encoder.encode(rows,batch_size=1))
    result=dict(viewVersion=VIEW_VERSION,encoderRevision=REVISION,queryEncodingBatchSize=1,
                identitySha256=hashlib.sha256('\n'.join(r['identity'] for r in rows).encode()).hexdigest(),
                calSplitSha256=provenance['splitHashes']['cal'],embeddingsSha256=file_sha(out/'embeddings.npy'),
                count=len(rows),estimatorFit=False,newTestRead=False)
    (out/'manifest.json').write_text(json.dumps(result,indent=2)); print('DETERMINISTIC CAL FROZEN',json.dumps(result),flush=True)


if __name__=='__main__':main()
