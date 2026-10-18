#!/usr/bin/env python3
"""First-use candidate CSV schema check: no message text committed or scored."""
import csv,hashlib,io,json,urllib.request
from pathlib import Path
from collections import Counter
SOURCE="sniff07/phishing-dataset"
REV="c21f524c014a49626dc61322a5f7b3194ac5c7d1"
FILES=("final_cleaned_safe_1.csv","final_cleaned_phis1.csv","final_cleaned_phis2.csv","final_cleaned_phis_3.csv","final_cleaned_phis_4.csv")
def fetch(name):
    if name not in FILES:raise ValueError("Unlisted source")
    u="https://raw.githubusercontent.com/"+SOURCE+"/"+REV+"/A%20Novel%20dataset/"+name
    with urllib.request.urlopen(urllib.request.Request(u,headers={"User-Agent":"MailGuard-v76-audit"}),timeout=90) as r:
        contents=r.read(25000001)
    if len(contents)>25000000:raise ValueError("Oversized CSV")
    return contents
def audit(raw,name):
    csv.field_size_limit(1000000)
    reader=csv.reader(io.StringIO(raw.decode("utf-8-sig","replace")))
    try:head=next(reader)
    except StopIteration:raise ValueError("Empty CSV")
    if not 1<=len(head)<=120 or any(len(s)>80 for s in head):raise ValueError("Invalid header")
    lengths=Counter()
    first=[]
    count=0
    for count,row in enumerate(reader,1):
        if count>100000:raise ValueError("Too many rows")
        lengths[len(row)]+=1
        if count<8:first.append([min(len(v),100000) for v in row])
    return dict(name=name,byteCount=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
      header=head,rowCount=count,rowWidths=dict(lengths),firstSevenCellLengthsOnly=first,
      provisionalClass="ham" if "safe" in name else "phishing")
def main():
    items=[]
    for name in FILES:
        a=audit(fetch(name),name);items.append(a)
        print("AUDIT",name,"headers",a["header"],"rows",a["rowCount"],"widths",a["rowWidths"],flush=True)
    result=dict(repo=SOURCE,revision=REV,licenseUnverified=True,originalityUnverified=True,
       textNotSaved=True,modelNotScored=True,files=items)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/v76-source-audit.json").write_text(json.dumps(result,indent=2))
if __name__=="__main__":main()
