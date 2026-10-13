import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from v61_semantic import windows,neighbours,script,training_pool


class SemanticIsolation(unittest.TestCase):
    def test_no_source_or_label_in_encoder_input(self):
        row=dict(focus='subject Meeting hosts example.invalid start Confirm the meeting',context='Confirm the meeting tomorrow',identity='x',source='secret-source',y=1)
        changed=dict(row,source='other',y=0)
        self.assertEqual(windows(row),windows(changed))
        self.assertNotIn('example.invalid',windows(row)[0])

    def test_japanese_does_not_get_ham_free_bucket(self):
        self.assertEqual(script(dict(context='打ち合わせの日程をご確認ください')),'cjk')
        self.assertEqual(script(dict(context='请确认会议时间和地点谢谢')),'cjk')

    def test_neighbour_gap_distinguishes_train_classes(self):
        vectors=np.array([[1.,0.],[0.,1.]],dtype=np.float32)
        near=neighbours(vectors,vectors,np.array([0,1]))
        self.assertLess(near[0,2],0.)
        self.assertGreater(near[1,2],0.)
        with self.assertRaises(ValueError): neighbours(vectors,vectors,np.array([1,1]))

    def test_training_cap_does_not_use_estimator_scores(self):
        rows=[dict(identity=str(i),source='SpamArchive-2026-04',y=1,score=i) for i in range(8)]
        altered=[dict(r,score=-r['score']) for r in rows]
        self.assertEqual([r['identity'] for r in training_pool(rows)],[r['identity'] for r in training_pool(altered)])


if __name__=='__main__':unittest.main()
