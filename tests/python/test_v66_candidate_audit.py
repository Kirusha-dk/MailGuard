import json,sys,tempfile,unittest,gzip,sqlite3
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from prepare_v66_candidate_audit import dedup,check_prior

def row(identity,label,alias):
    return dict(identity=identity,label=label,source='fixture',family=identity,aliases=[alias])

class CandidateAudit(unittest.TestCase):
    def test_duplicate_and_conflicting_labels_are_not_counted_as_new(self):
        a=[row('one',1,'body:x'),row('two',1,'body:x'),row('three',0,'body:y')]
        out,stats=dedup(a)
        self.assertEqual(len(out),1)
        self.assertEqual(stats['duplicate'],1)
        self.assertEqual(out[0]['label'],0)
        out,stats=dedup([row('one',1,'body:x'),row('two',0,'body:x')])
        self.assertFalse(out)
        self.assertEqual(stats['conflictingLabel'],1)

    def test_prior_block_and_all_previous_test_shards_required(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            db=root/'db.sqlite'
            con=sqlite3.connect(db)
            con.execute('CREATE TABLE keys(key TEXT PRIMARY KEY)')
            con.execute('INSERT INTO keys VALUES (?)',('body:old',))
            con.commit();con.close()
            for n in range(5):
                with gzip.open(root/f'input-{n}.jsonl.gz','wt') as f:
                    f.write(json.dumps(dict(identity='prior',family='prior',
                         focus='subject Hi hosts  start X',context='subject Hi body Y'))+'\n')
            with self.assertRaisesRegex(ValueError,'Incomplete old test shard'):
                check_prior([row('a',1,'body:old')],db,root)

if __name__=='__main__':unittest.main()
