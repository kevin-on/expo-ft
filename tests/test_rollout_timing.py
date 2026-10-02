"""Clock-domain, source and UI parity checks; no SDK or device access."""
import io
from contextlib import redirect_stdout
from pathlib import Path
import runpy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from expo_ft.env.robot_eval import RobotEvaluation
from expo_ft.env.rollout_timing import RolloutMetrics, format_rollout_metrics, step_timing

ROOT = Path(__file__).resolve().parents[1]
Dashboard = runpy.run_path(str(ROOT/'expo_ft/utils/rollout_dashboard.py'))['RolloutDashboard']


def timing(source='policy', completed=10., ages=None):
    env = SimpleNamespace(
        get_action_timing=lambda: dict(ws_action_send_ms=99999999.,
            frame_age_ms={'side':120., 'wrist':125.} if ages is None else ages),
        get_observation_timing=lambda: dict(rpc_ms=30., ws={'side_frame_age_ms':7.}))
    return step_timing(env, step=2, source=source, observed=completed-.2,
        policy_observed=completed-.4, dispatched=completed-.05,
        acted=completed-.03, arrived=completed, plan_ms=12.)


class TimingTests(unittest.TestCase):
    def test_intervals_and_ages_use_separate_clocks_and_observations(self):
        t = timing()
        self.assertAlmostEqual(t['action_rpc_ms'],20.)
        self.assertAlmostEqual(t['observation_ms'],30.)
        self.assertAlmostEqual(t['observation_age_ms'],150.)
        self.assertAlmostEqual(t['policy_observation_age_ms'],350.)
        metrics = RolloutMetrics()
        metrics.step(timing=t)
        self.assertEqual(metrics.snapshot(active=True,now=10)['frame_age_ms'],{'side':120.,'wrist':125.})
        # Missing/human ages must not reuse the previous policy measurement.
        metrics.step(timing=timing('human',completed=10.1))
        state = metrics.snapshot(active=True,now=10.1)
        self.assertAlmostEqual(state['rollout_hz'],10.)
        self.assertEqual(state['frame_age_ms'],{})
        self.assertIsNone(timing('human')['policy_observation_age_ms'])
        metrics.step(timing=timing(ages={'side':float('nan'),'wrist':-1}))
        self.assertEqual(metrics.snapshot(active=True)['frame_age_ms'],{})
        metrics.step(timing=timing(ages={}))
        self.assertEqual(metrics.frame_age_ms,{})
        self.assertEqual(format_rollout_metrics(metrics.snapshot(active=False)),
                         '— Hz | Frame age S/W: —/— ms')
        metrics.reset()
        self.assertIsNone(metrics.hz(100))
        self.assertEqual(metrics.frame_age_ms,{})

    def test_eval_and_online_show_same_completed_step_then_hide_on_reset(self):
        env=SimpleNamespace(close=lambda:None)
        session=RobotEvaluation({0:env},lambda _:None,replan_steps=1,control_hz=10,max_steps=80)
        ui=Dashboard(1,80)
        try:
            ui.ready(0,0)
            session.state(0,status='starting')
            for i in range(2):
                t=timing(completed=10+i*.1)
                ui.step(0,i+1,False,timing=t)
                session.state(0,status='running',steps=i+1,timing=t)
            with patch('expo_ft.env.rollout_timing.time.monotonic',return_value=10.1):
                expected=format_rollout_metrics(session.snapshot()[0])
                self.assertEqual(expected,'10.0 Hz | Frame age S/W: 120/125 ms')
                output=io.StringIO()
                with redirect_stdout(output):ui.draw()
                self.assertIn(expected,output.getvalue())
            session.state(0,status='resetting')
            ui.resetting(0)
            expected=format_rollout_metrics(session.snapshot()[0])
            self.assertEqual(expected,'— Hz | Frame age S/W: —/— ms')
            output=io.StringIO()
            with redirect_stdout(output):ui.draw()
            self.assertIn(expected,output.getvalue())
        finally:session.close()

    def test_direct_eval_uses_same_metric_labels(self):
        display=runpy.run_path(str(ROOT/'eval_sft_robots.py'))['display']
        session=SimpleNamespace(max_steps=80,snapshot=lambda:{1:dict(
            status='running',steps=5,episodes=1,successes=1,last=True,
            rollout_hz=9.8,frame_age_ms={'side':120.,'wrist':125.})})
        output=io.StringIO()
        with redirect_stdout(output):display(session,30)
        self.assertIn('9.8 Hz | Frame age S/W: 120/125 ms',output.getvalue())


if __name__ == '__main__':
    unittest.main()
