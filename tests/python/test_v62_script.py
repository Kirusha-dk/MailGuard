import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from train_v55_multilingual import views
from v62_runtime import features,FEATURES


class ScriptFeatureIsolation(unittest.TestCase):
    def test_label_and_source_do_not_change_features(self):
        row=dict(zip(('focus','context'),views('SUBJECT: Meeting\nBODY: Please confirm the meeting tomorrow.')))
        a=features([dict(row,y=0,source='ham-source')])
        b=features([dict(row,y=1,source='spam-source')])
        self.assertEqual(a.shape,(1,FEATURES))
        self.assertEqual((a!=b).nnz,0)

    def test_caller_batch_does_not_change_character_features(self):
        rows=[dict(zip(('focus','context'),views(t))) for t in
              ['SUBJECT: Meeting\nBODY: Please confirm tomorrow.', 'SUBJECT: 会议\nBODY: 请确认会议时间谢谢']]
        batch=features(rows)
        for i,row in enumerate(rows):self.assertEqual((batch[i]!=features([row])).nnz,0)


if __name__=='__main__':unittest.main()
