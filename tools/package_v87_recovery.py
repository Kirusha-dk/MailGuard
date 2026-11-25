"""Publish only the verified retained adapter; never execute training here."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

SOURCE_ZIP='676c5134eb0c25b6ebd8bf5c0c8d91c8c480fda853824586f4b8baa2815c04ee'
BOOTSTRAP_ZIP='c79b946223ff66c704d72d129c8316458c41c35bf11ad7bf70cdaf1372b40355'
PARENT='2e62947281989573c47e322e2dd372a9d0690750f4490c9238f9f962ef17e685'
SOURCE='56e8284f958dbe59bc7c90c590dbc3458d72fdaae31a4e6b78e6d67bd88a5b47'

def digest(b):return hashlib.sha256(b).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--bootstrap',type=Path,required=True)
    p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--backbone',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    recovery=json.loads((a.root/'reports/v87-repair/2026-11-24/recovery.json').read_text())
    assert recovery['selection']['selected']=='retained-parent','Candidate weights must not be substituted'
    assert digest(a.source.read_bytes())==SOURCE_ZIP
    assert digest(a.bootstrap.read_bytes())==BOOTSTRAP_ZIP
    members={}
    with zipfile.ZipFile(a.source) as z:
        members['selected-adapter.safetensors']=z.read('v86-state/best-adapter.safetensors')
        members['selected-development.json']=z.read('v86-state/best-development.json')
        members['paused-v86/checkpoint.pt']=z.read('v86-state/checkpoint.pt')
        backbone=json.loads(z.read('.cache/v82/backbone/manifest.json'))
    with zipfile.ZipFile(a.bootstrap) as z:members['retained-parent/checkpoint.pt']=z.read('parent-v84-step46000.pt')
    assert digest(members['paused-v86/checkpoint.pt'])==SOURCE
    assert digest(members['retained-parent/checkpoint.pt'])==PARENT
    assert digest(members['selected-adapter.safetensors'])==recovery['selected_adapter_sha256']
    assert json.loads(members['selected-development.json'])==recovery['selected_development']
    for name,expected in backbone['files'].items():
        if a.backbone:
            b=(a.backbone/name).read_bytes()
        else:
            url=f"https://huggingface.co/{backbone['model']}/resolve/{backbone['revision']}/{name}"
            with urllib.request.urlopen(url,timeout=180) as response:b=response.read()
        assert digest(b)==expected,('Backbone drift',name)
        members['backbone/'+name]=b
    members['backbone/manifest.json']=(json.dumps(backbone,indent=2)+'\n').encode()
    paths=['docs/v87-english-repair.md','docs/v86-english-month.md','docs/v84-source-attribution.md',
           'research/v86/frozen.json','research/v86/archive-sources.json',
           'reports/v87-repair/2026-11-24/recovery.json',
           'reports/v87-repair/2026-11-24/regression.json',
           'reports/v87-repair/2026-11-24/guard-verification.json',
           'reports/v87-repair/2026-11-24/trial-verification.json',
           'reports/v87-repair/2026-11-24/trial-manifest.json']
    for path in paths:members[path]=(a.root/path).read_bytes()
    metadata=dict(experiment='v87-en-repair-20261124',selected_weights_step=46000,
        paused_training_checkpoint_step=49500,weights_and_training_checkpoints_are_separate=True,
        selected_for_inference_only=True,automatic_training_resume_allowed=False,
        accepted_for_production=False,optimizer_steps_in_publication=0,
        source_zip_sha256=SOURCE_ZIP,bootstrap_zip_sha256=BOOTSTRAP_ZIP,
        selected_adapter_sha256=recovery['selected_adapter_sha256'],raw_mail_included=False,
        pinned_backbone_weights_included=True,attribution_and_source_conditions_preserved=True)
    members['metadata.json']=(json.dumps(metadata,indent=2)+'\n').encode()
    members['checksums.json']=(json.dumps({n:digest(b) for n,b in sorted(members.items())},indent=2)+'\n').encode()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.output,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,b in sorted(members.items()):
            info=zipfile.ZipInfo(name,(2026,11,24,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16
            z.writestr(info,b)
    with zipfile.ZipFile(a.output) as z:
        checks=json.loads(z.read('checksums.json'))
        assert set(z.namelist())==set(checks)|{'checksums.json'}
        assert all(digest(z.read(n))==expected for n,expected in checks.items())
    print(json.dumps(dict(package_sha256=digest(a.output.read_bytes()),bytes=a.output.stat().st_size,members=len(members),optimizer_steps=0)))
if __name__=='__main__':main()
