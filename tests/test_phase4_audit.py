"""Regression checks for train-only analysis policy and honest denominators."""
import importlib.util
from pathlib import Path
import unittest
import tempfile

import pandas as pd

spec = importlib.util.spec_from_file_location('audit_phase4', Path(__file__).resolve().parents[1]/'scripts/audit_phase4.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class Phase4AuditTests(unittest.TestCase):
    def test_csv_float_roundtrip_preserves_tie_order(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'m.csv'
            p.write_text('data_split,regularized_curvature_energy_inv_unit\ntrain,0.11708934307252043\ntrain,0.11708934307252042\n',encoding='utf-8')
            f=audit.read_train(p)
            self.assertGreater(f.iloc[0,1],f.iloc[1,1])
            self.assertEqual(f.iloc[0,1],float('0.11708934307252043'))

    def test_holdout_rejected_by_research_entry(self):
        for split in ['validation','test','challenge']:
            with self.assertRaises(ValueError):
                audit.train_only(pd.DataFrame({'data_split':['train',split]}))

    def test_absolute_tolerance_handles_zero(self):
        self.assertTrue(audit.stability(1e-7,0.,0.,'L'))
        self.assertFalse(audit.stability(2e-6,0.,0.,'L'))
        self.assertIsNone(audit.stability(None,0.,0.,'L'))

    def test_both_resolution_checks_required(self):
        self.assertFalse(audit.stability(1.,1.2,1.,'L'))
        self.assertFalse(audit.stability(1.2,1.,1.,'L'))
        self.assertTrue(audit.stability(1.001,1.001,1.,'L'))

    def test_no_valid_classification_is_exclusive(self):
        def frame(reach,collision,valid=None):
            return pd.DataFrame({'fully_reachable':reach,'collision_free':collision,'selection_valid':valid or [False]*len(reach)})
        self.assertEqual(audit.no_valid_reason(frame([False,False],[None,None])),'all_unreachable')
        self.assertEqual(audit.no_valid_reason(frame([True,True],[False,False])),'all_reachable_colliding')
        self.assertEqual(audit.no_valid_reason(frame([False,True],[None,False])),'mixed_unreachable_collision')
        self.assertEqual(audit.no_valid_reason(frame([True],[None])),'other_calculation_or_schema')
        self.assertEqual(audit.no_valid_reason(frame([True],[True],[True])),'has_valid')

    def test_tie_handling_matches_generator_tolerance(self):
        rows=[]
        for i,value in enumerate([1.,1.+5e-13,1.1]):
            row=dict(data_split='train',scenario_id='s',run_id=str(i),curve_key='line',shape=0.,elbow_mode='up',selection_valid=True)
            row.update({field:value for field in audit.OBJECTIVES.values()})
            rows.append(row)
        w=audit.winner_rows(pd.DataFrame(rows))
        self.assertTrue(w.tie_count.eq(2).all())
        self.assertTrue(w.winner_run_id.eq('0').all())
        self.assertTrue(w.tied_run_ids_json.eq('["0", "1"]').all())

    def test_no_valid_labels_are_not_wins(self):
        row=dict(data_split='train',scenario_id='s',selection_valid=False)
        w=audit.winner_rows(pd.DataFrame([row]))
        self.assertEqual(len(w),6)
        self.assertTrue(w.winner_run_id.eq('').all())
        self.assertTrue(w.tie_count.eq(0).all())

if __name__=='__main__':unittest.main()
