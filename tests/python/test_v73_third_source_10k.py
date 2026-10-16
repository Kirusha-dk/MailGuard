import sys,unittest,heapq
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"tools"))
from v73_third_source_10k import (lexical_cohort_filter,source_screen,keep_top,
                                  fingerprint,rank,SAMPLE,HEAP_CAP)
import v68_ru_en_research as v68

class NewRepublisherSafetyChecks(unittest.TestCase):
    def test_language_and_length_screen(self):
        self.assertIsNone(lexical_cohort_filter("Привет. Сегодня отличный день для работы и командных встреч"))
        self.assertIsNone(lexical_cohort_filter("short"))
        self.assertIsNotNone(lexical_cohort_filter("Your company invoice notification has been delivered securely."))
    def test_overlap_precedes_scoring(self):
        c="Please verify the payment invoice for business accounting department"
        i=v68.digest(c)
        f=v68.digest(c[:160])
        self.assertEqual(source_screen(c,{i},set())[0],"exact_prior")
        self.assertEqual(source_screen(c,set(),{f})[0],"prefix_prior")
        self.assertEqual(source_screen(c,set(),set())[0],"eligible")
    def test_fixed_label_blind_heap(self):
        data=[(f"{i:064x}",f"{i:064x}",f"Example customer mail {i}",i%2) for i in range(73)]
        def sample(reverse_labels):
            heap=[]
            for i,f,t,y in data:
                keep_top(heap,i,f,t,1-y if reverse_labels else y,maxkeep=25)
            return sorted([row["identity"] for _,_,row in heap])
        self.assertEqual(sample(False),sample(True))
        self.assertEqual(len(sample(False)),25)
        self.assertEqual(SAMPLE,10000)
        self.assertGreaterEqual(HEAP_CAP,SAMPLE)
    def test_fingerprint_ignores_labels(self):
        a=[dict(identity="a",family="f",y=1),dict(identity="b",family="g",y=0)]
        digest=fingerprint(a)
        for r in a:r["y"]=1-r["y"]
        self.assertEqual(fingerprint(a),digest)
        self.assertEqual(len(digest),64)
if __name__=="__main__":unittest.main()
