"""Tiny stdlib-only metric checks; no JAX or hardware imports."""
import json
import unittest

from test_split_reset_overlap_isolated import load_definitions

namespace = {}
load_definitions('expo_ft/utils/log_utils.py', ['InterventionStats'], namespace)
InterventionStats = namespace['InterventionStats']


class InterventionStatsTests(unittest.TestCase):
    def test_weighted_round_rate_and_episode_counts_survive_resume(self):
        stats = InterventionStats(2)
        metrics = {}
        stats.on_round_done([
            ([{'is_hil': True}, {'is_hil': True}], False),
            ([{'is_hil': False}] * 6, True),
        ], metrics)
        self.assertEqual(metrics['training/intervention_rate'], 0.25)
        self.assertEqual(metrics['robot-0/intervention_rate'], 1.0)
        self.assertEqual(metrics['robot-1/intervention_rate'], 0.0)
        self.assertEqual(metrics['training/episodes_with_intervention'], 1)
        self.assertEqual(metrics['training/total_intervention_transitions'], 2)
        saved = json.loads(json.dumps(stats.state_dict()))
        resumed = InterventionStats(2, saved)
        resumed.on_round_done([
            ([{'is_hil': False}], True), ([{'is_hil': True}], False),
        ], metrics)
        self.assertEqual(metrics['training/intervention_rate'], 0.5)
        self.assertEqual(metrics['training/episodes_with_intervention'], 2)
        self.assertEqual(metrics['training/total_intervention_transitions'], 3)
        self.assertEqual(metrics['robot-0/total_intervention_transitions'], 2)
        self.assertEqual(metrics['robot-1/episodes_with_intervention'], 1)
        self.assertEqual(saved, stats.state_dict())

    def test_old_checkpoint_and_no_interventions(self):
        stats = InterventionStats(2, None)
        metrics = {}
        stats.on_round_done([([{'is_hil': False}], False), ([], False)], metrics)
        self.assertTrue(all(value == 0 for value in metrics.values()))


if __name__ == '__main__':
    unittest.main()
