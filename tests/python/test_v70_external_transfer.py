import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"tools"))
from v70_external_transfer import (raw_to_text,calibrate,cal_cap,filter_prior,
                                   freeze_source,ARCHIVES,SOURCE_REV,CANDIDATES)

class ExternalFirstUseTransferChecks(unittest.TestCase):
    def test_mime_plain_and_subject(self):
        data=(b"From: sender@example.test\r\nSubject: Meeting notice\r\n"
              b"MIME-Version: 1.0\r\nContent-Type: text/plain; charset=utf-8\r\n\r\n"
              b"Please schedule our next team meeting on Friday.")
        parsed=raw_to_text(data)
        self.assertIn("Meeting notice",parsed)
        self.assertIn("team meeting",parsed)
    def test_html_hides_script(self):
        raw=(b"Subject: A\r\nContent-Type: text/html; charset=utf-8\r\n\r\n"
             b"<p>Invoice</p><script>maliciousNoise</script><div>Verify details</div>")
        txt=raw_to_text(raw)
        self.assertNotIn("maliciousNoise",txt)
        self.assertIn("Invoice",txt)
        self.assertIn("Verify details",txt)
    def test_cal_only_candidate_uses_ham_counts(self):
        n_ham=11000
        truth=np.array([0]*n_ham+[1]*1500)
        score=np.r_[np.linspace(-2,1,n_ham),np.linspace(-.5,2,1500)]
        pool={k:score*(1+i*.01) for i,k in enumerate(CANDIDATES)}
        policy=calibrate(pool,truth,0)
        self.assertEqual(policy["calFP"],0)
        self.assertEqual(policy["calHAM"],n_ham)
        self.assertIn(policy["candidate"],CANDIDATES)
        relaxed=calibrate(pool,truth,cal_cap(n_ham,"sensitivity-0p05pct-cal-fp"))
        self.assertLessEqual(relaxed["calFP"],5)
        self.assertGreaterEqual(relaxed["calTP"],policy["calTP"])
    def test_prior_content_overlap(self):
        rows=[{"identity":"a","family":"f","y":0},
              {"identity":"b","family":"g","y":1},
              {"identity":"c","family":"h","y":1}]
        selected,excluded=filter_prior(rows,{"a"},{"g"})
        self.assertEqual([r["identity"] for r in selected],["c"])
        self.assertEqual(excluded["exact_content_overlap"],1)
        self.assertEqual(excluded["first160_character_overlap"],1)
    def test_freeze_metadata_excludes_mail_text_and_labels(self):
        rows=[{"identity":"a","family":"f","y":0,"text":"DO NOT PERSIST"},
              {"identity":"b","family":"g","y":1,"text":"DO NOT PERSIST"}]
        obj=freeze_source(rows,{"ham.zip":"abc","spam.zip":"def"},{},{})
        self.assertEqual(obj["eligibleCount"],2)
        self.assertNotIn("DO NOT PERSIST",str(obj))
        self.assertEqual(len(obj["identityFamilyFingerprint"]),64)
        self.assertEqual(len(SOURCE_REV),40)
        self.assertEqual(set(ARCHIVES),{"ham.zip","spam.zip"})

if __name__=="__main__":unittest.main()
