import unittest
import sys
from pathlib import Path
import tempfile
import json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from robot_planner.release_io import iter_release_metrics


class ReleaseIOTests(unittest.TestCase):
    def test_heldout_closed_by_default(self):
        for split in ('validation','test','challenge'):
            with self.assertRaises(PermissionError):list(iter_release_metrics('not-read',split=split))

    def test_incomplete_dataset_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder)/'generation_state.json').write_text(json.dumps({'status':'running'}))
            with self.assertRaises(ValueError):list(iter_release_metrics(folder))


if __name__=='__main__':unittest.main()
