import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"tools"))
from v76_external_nn_firstuse import (
  classify_label, canonical_row, select_fixed,screen,paired,fingerprint,COHORT,
  EXPECTED)
class V76FirstUseTests(unittest.TestCase):
    def test_explicit_label_semantics(self):
        a={"category_numeric":"0.0","category_text":"ham","subject":"Hello","body":"normal office communication about monthly invoice and payment"}
        b={"category_numeric":"1","category_text":"spam","subject":"Verify","body":"you must urgently verify your bank account on suspicious website"}
        self.assertEqual(classify_label(a,"final_cleaned_safe_1.csv")[0],0)
        self.assertEqual(classify_label(b,"final_cleaned_phis1.csv")[0],1)
        with self.assertRaises(ValueError):classify_label(b,"final_cleaned_safe_1.csv")
        with self.assertRaises(ValueError):classify_label({**b,"category_text":"undefined"},"final_cleaned_phis1.csv")
    def test_hash_sampling_never_reads_labels(self):
        rows=[{"identity":f"{i:064x}","family":f"{i:064x}","y":i%2} for i in range(100)]
        a=[x["identity"] for x in select_fixed(rows,30)]
        for row in rows:row["y"]=1-row["y"]
        b=[x["identity"] for x in select_fixed(list(reversed(rows)),30)]
        self.assertEqual(a,b)
    def test_prior_screen_excludes_exact_and_family(self):
        rows=[dict(identity="x",family="A"),dict(identity="y",family="B"),
              dict(identity="z",family="C")]
        allowed,why=screen(rows,{"x"},{"B"})
        self.assertEqual([x["identity"] for x in allowed],["z"])
        self.assertEqual(why["exact_prior"],1)
        self.assertEqual(why["prefix_prior"],1)
    def test_paired_only_counts_transitions(self):
        outcomes=paired([True,False,True,False],[True,True,False,False],[1,1,0,0])
        self.assertEqual(outcomes["spam_both_positive"],1)
        self.assertEqual(outcomes["spam_new_only"],1)
        self.assertEqual(outcomes["ham_old_only"],1)
        self.assertEqual(outcomes["ham_both_negative"],1)
    def test_source_row_is_canonical_and_pinned(self):
        row,reason=canonical_row({
           "body":"Office update about our quarterly calendar with urgent invoice questions",
           "subject":"Scheduling questions","category_text":"ham","category_numeric":"0.0"
        },"final_cleaned_safe_1.csv")
        self.assertIsNone(reason)
        self.assertEqual(row["y"],0)
        self.assertEqual(len(row["identity"]),64)
        self.assertEqual(len(EXPECTED),5)
        self.assertEqual(COHORT,9000)
if __name__=="__main__":unittest.main()
