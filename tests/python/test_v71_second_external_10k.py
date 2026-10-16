import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"tools"))
from v71_second_external_10k import (read_dataset,select_fixed,fingerprint,SAMPLE)
import io,tarfile
def tarbytes(rows):
    blob=io.BytesIO()
    with tarfile.open(fileobj=blob,mode="w:gz") as t:
        for filename,content in rows:
            b=content.encode("ascii")
            tar=tarfile.TarInfo(filename)
            tar.size=len(b)
            t.addfile(tar,io.BytesIO(b))
    return blob.getvalue()
class ExternalSecondProviderTests(unittest.TestCase):
    def test_tar_folders_labels_and_dedup(self):
        fake=tarbytes([
          ("email-dataset-fixed/dataset/1/a.txt","This is a normal team meeting notice for Friday."),
          ("email-dataset-fixed/dataset/2/b.txt","Claim a winner prize immediately by verifying your bank account."),
          ("email-dataset-fixed/dataset/1/c.txt","This is a normal team meeting notice for Friday.")
        ])
        with self.assertRaisesRegex(ValueError,"Not enough source emails"):
            read_dataset(fake)
    def test_fixed_selection_ignores_labels(self):
        original=[{"identity":f"{i:064x}","family":f"{i:064x}","y":i%2} for i in range(45)]
        a=[r["identity"] for r in select_fixed(original,20)]
        for row in original:row["y"]=1-row["y"]
        b=[r["identity"] for r in select_fixed(list(reversed(original)),20)]
        self.assertEqual(a,b)
        self.assertEqual(SAMPLE,10000)
    def test_fingerprint_no_labels(self):
        rows=[dict(identity="aaa",family="bbb",y=1),dict(identity="ccc",family="ddd",y=0)]
        before=fingerprint(rows)
        for r in rows:r["y"]=1-r["y"]
        self.assertEqual(before,fingerprint(rows))
        self.assertEqual(len(before),64)
if __name__=="__main__":unittest.main()
