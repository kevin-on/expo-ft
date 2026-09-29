"""Tiny stdlib-only metric checks; no JAX or hardware imports."""
from collections import deque
from dataclasses import dataclass, field
from types import SimpleNamespace
import unittest

from test_split_reset_overlap_isolated import load_definitions

namespace = dict(dataclass=dataclass, field=field, deque=deque)
load_definitions('expo_ft/utils/log_utils.py',
                 ['log_round_interventions', 'EpisodeState', 'TrainingStats'], namespace)
log_round_interventions = namespace['log_round_interventions']
TrainingStats = namespace['TrainingStats']
REMOVED = {'intervention_step_rate', 'intervention_episode_rate',
           'episodes_with_intervention', 'total_intervention_transitions'}


class InterventionStatsTests(unittest.TestCase):
    def assert_no_removed_metrics(self, metrics):
        self.assertFalse(REMOVED.intersection(k.split('/')[-1] for k in metrics))

    def test_weighted_step_rate_and_current_round_episode_mean(self):
        metrics = {}
        log_round_interventions([
            ([{'is_hil': True}] * 2, True),
            ([{'is_hil': False}] * 6, True),
        ], metrics)
        self.assertEqual(metrics['training/intervention_rate'], 0.25)
        self.assertEqual(metrics['robot-0/intervention_rate'], 1.0)
        self.assertEqual(metrics['robot-1/intervention_rate'], 0.0)
        self.assertEqual(metrics['training/had_intervention'], 0.5)
        self.assertEqual(metrics['training/success_without_intervention'], 0.5)
        self.assert_no_removed_metrics(metrics)
        # A later round must not inherit any interventions from this round.
        log_round_interventions([([{}], True), ([{}], True)], metrics)
        self.assertEqual(metrics['training/intervention_rate'], 0.0)
        self.assertEqual(metrics['training/had_intervention'], 0.0)
        self.assertEqual(metrics['training/success_without_intervention'], 1.0)

    def test_binary_indicators_match_success_and_any_human_step(self):
        for success in (False, True):
            for human_steps in (0, 1, 5):
                with self.subTest(success=success, human_steps=human_steps):
                    metrics = {}
                    records = [{'is_hil': i < human_steps} for i in range(6)]
                    log_round_interventions([(records, success)], metrics)
                    expected = float(success and human_steps == 0)
                    for prefix in ('robot-0', 'training'):
                        self.assertEqual(metrics[f'{prefix}/had_intervention'], float(human_steps > 0))
                        self.assertEqual(metrics[f'{prefix}/success_without_intervention'], expected)
                    self.assert_no_removed_metrics(metrics)

    def test_empty_round_and_empty_failed_episode(self):
        for episodes in ([], [([], False)]):
            metrics = {}
            log_round_interventions(episodes, metrics)
            self.assertTrue(all(value == 0 for value in metrics.values()))

    def test_single_robot_matches_round_metrics_without_accumulation(self):
        stats = TrainingStats()
        for success, human_steps in ((True, 1), (True, 0), (False, 0), (False, 3)):
            ep = SimpleNamespace(human_steps=human_steps, policy_steps=6-human_steps,
                                 ep_return=float(success), ep_len=6)
            metrics = {}
            stats.on_episode_done(ep, success, metrics)
            expected = {}
            log_round_interventions([
                ([{'is_hil': i < human_steps} for i in range(6)], success),
            ], expected)
            for key in ('intervention_rate', 'had_intervention', 'success_without_intervention'):
                self.assertEqual(metrics[f'training/{key}'], expected[f'training/{key}'])
            self.assertEqual(metrics['training/success'], float(success))
            self.assert_no_removed_metrics(metrics)


if __name__ == '__main__':
    unittest.main()
