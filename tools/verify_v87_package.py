"""Verify an already published retained package without rebuilding or gradients."""
import argparse,hashlib,json,zipfile
from pathlib import Path

def digest(b):return hashlib.sha256(b).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--recovery',type=Path,required=True);a=p.parse_args();r=json.loads(a.recovery.read_text())
 with zipfile.ZipFile(a.package) as z:
  checks=json.loads(z.read('checksums.json'));assert set(z.namelist())==set(checks)|{'checksums.json'}
  assert len(checks)==22
  for name,h in checks.items():assert digest(z.read(name))==h,name
  assert digest(z.read('selected-adapter.safetensors'))==r['selected_adapter_sha256']
  assert digest(z.read('retained-parent/checkpoint.pt'))=='2e62947281989573c47e322e2dd372a9d0690750f4490c9238f9f962ef17e685'
  assert digest(z.read('paused-v86/checkpoint.pt'))=='56e8284f958dbe59bc7c90c590dbc3458d72fdaae31a4e6b78e6d67bd88a5b47'
  assert json.loads(z.read('reports/v87-repair/2026-11-24/recovery.json'))==r
  assert json.loads(z.read('selected-development.json'))==r['selected_development']
  b=json.loads(z.read('backbone/manifest.json'));assert b['revision']=='1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
  for name,h in b['files'].items():assert digest(z.read('backbone/'+name))==h
  meta=json.loads(z.read('metadata.json'));assert meta['selected_for_inference_only'] and not meta['automatic_training_resume_allowed'] and not meta['raw_mail_included']
 print(json.dumps(dict(package_sha256=digest(a.package.read_bytes()),bytes=a.package.stat().st_size,members=len(checks)+1,optimizer_steps=0,existing_package_verified=True)))
if __name__=='__main__':main()
