import sys
from pathlib import Path
import unittest
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_phase4_v3 import classify,candidate_winner,add_bins,hashed


def candidate(status='accepted',u=False,c=False):
    return dict(selection_valid=status=='accepted',candidate_status=status,has_unreachable_witness=u,has_collision_witness=c)


class AnalysisTests(unittest.TestCase):
    def test_unresolved_alternative_excludes_entire_scenario(self):
        r=classify([candidate(),candidate('unresolved')])
        self.assertEqual(r['cohort'],'excluded_unresolved');self.assertEqual(r['accepted'],1)

    def test_no_valid_not_environment_impossibility(self):
        r=classify([candidate('invalid',u=True),candidate('invalid',c=True)])
        self.assertEqual(r['reason'],'mixed_unreachable_collision');self.assertEqual(r['cohort'],'excluded_no_valid')

    def test_unresolved_without_valid_not_confirmed_failure(self):
        self.assertEqual(classify([candidate('unresolved')])['reason'],'unresolved_or_calculation')

    def test_ties_preserve_raw_sort_before_lexical(self):
        a=dict(cost=1.,path_length_units=2.,curve_key='line',shape=0.,elbow_mode='up',run_id='a')
        b=dict(a,cost=1.-1e-13,run_id='z')
        w,t=candidate_winner([a,b],'cost');self.assertEqual(w['run_id'],'z');self.assertEqual(len(t),2)

    def test_exact_bin_boundaries(self):
        s=pd.DataFrame([dict(link1=.75,link2=1.,chord_length=.875,angular_separation_deg=60.,start_radius=.7,goal_radius=1.4)])
        r=add_bins(s).iloc[0]
        self.assertTrue(r.ratio_bin.startswith('[0.75,'));self.assertTrue(r.motion_bin.startswith('[0.5,'))
        self.assertTrue(r.angle_bin.startswith('[60,'))

    def test_hash_selection_is_deterministic(self):
        self.assertEqual(hashed('run1'),hashed('run1'));self.assertNotEqual(hashed('run1'),hashed('run2'))


if __name__=='__main__':unittest.main()
