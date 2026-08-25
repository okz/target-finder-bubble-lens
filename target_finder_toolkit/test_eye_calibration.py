import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from target_finder_toolkit.eye_calibration import EyeCalibration


class _FakeTracker:
    def __init__(self):
        self.affine_matrix = np.ones((2, 3), dtype=np.float64)
        self.affine_matrix_tf = object()
        self.kalman_filter = SimpleNamespace(
            x=np.ones((4, 1), dtype=np.float64),
            P=np.full((4, 4), 2.0, dtype=np.float64),
        )
        self.adapt_called = False
        self.adapt_kwargs = None

    def adapt_from_gaze_results(self, gaze_results, norm_pogs, **kwargs):
        # No infer_fn on this fake tracker, so EyeCalibration falls back to
        # the pre-adaptation norm_pog values it already collected -- this
        # fake only needs to record that adaptation was requested and with
        # what hyperparameters, not actually change any model weights.
        self.adapt_called = True
        self.adapt_kwargs = kwargs
        self.adapt_sample_count = len(gaze_results)


class EyeCalibrationTest(unittest.TestCase):
    def test_fit_adapts_model_then_initializes_manual_correction(self):
        screen_w, screen_h = 1000, 800
        done = []
        calibration = EyeCalibration(
            screen_w,
            screen_h,
            num_points=5,
            on_done=lambda success, error: done.append((success, error)),
        )
        tracker = _FakeTracker()
        calibration.start(tracker)

        # Build raw samples whose known affine mapping reaches every target.
        expected_gain_x = 1.2
        expected_gain_y = 0.8
        expected_bias_x = -0.05
        expected_bias_y = 0.04
        calibration._gaze_results = []
        for target_x, target_y in calibration.targets:
            target_norm_x = target_x / screen_w - 0.5
            target_norm_y = target_y / screen_h - 0.5
            raw_x = (target_norm_x - expected_bias_x) / expected_gain_x
            raw_y = (target_norm_y - expected_bias_y) / expected_gain_y
            calibration._gaze_results.append(
                [SimpleNamespace(norm_pog=np.array([raw_x, raw_y])) for _ in range(5)]
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            calibration.SAVE_DIR = Path(tmpdir)
            calibration._fit()

        self.assertTrue(tracker.adapt_called)
        self.assertEqual(tracker.adapt_kwargs["steps_inner"], EyeCalibration.MAML_STEPS_INNER)
        self.assertEqual(tracker.adapt_kwargs["inner_lr"], EyeCalibration.MAML_INNER_LR)
        self.assertFalse(tracker.adapt_kwargs["affine_transform"])
        self.assertTrue(calibration.is_calibrated)
        self.assertIsNone(tracker.affine_matrix)
        self.assertIsNone(tracker.affine_matrix_tf)
        self.assertTrue(np.allclose(tracker.kalman_filter.x, 0.0))
        self.assertTrue(np.allclose(tracker.kalman_filter.P, np.eye(4)))
        self.assertEqual(len(done), 1)
        self.assertTrue(done[0][0])
        self.assertLess(done[0][1], 1e-6)

        correction = calibration.correction_values
        self.assertAlmostEqual(correction["gaze_gain_x"], 1.0, places=6)
        self.assertAlmostEqual(correction["gaze_gain_y"], 1.0, places=6)
        self.assertAlmostEqual(correction["gaze_offset_x"], 0.0, places=6)
        self.assertAlmostEqual(correction["gaze_offset_y"], 0.0, places=6)
        affine = np.array(correction["rake_affine_matrix"], dtype=np.float64)
        self.assertAlmostEqual(affine[0, 0], expected_gain_x, places=6)
        self.assertAlmostEqual(affine[1, 1], expected_gain_y, places=6)
        self.assertAlmostEqual(affine[0, 2], expected_bias_x, places=6)
        self.assertAlmostEqual(affine[1, 2], expected_bias_y, places=6)

    def test_fit_rejects_bad_runtime_manual_model_when_affine_has_shear(self):
        screen_w, screen_h = 1000, 800
        done = []
        calibration = EyeCalibration(
            screen_w,
            screen_h,
            num_points=5,
            on_done=lambda success, error: done.append((success, error)),
        )
        tracker = _FakeTracker()
        calibration.start(tracker)

        raw_points = [
            (-0.42, -0.35),
            (0.38, -0.30),
            (0.0, 0.0),
            (-0.37, 0.34),
            (0.41, 0.32),
        ]
        calibration._gaze_results = []
        for (target_x, target_y), (raw_x, raw_y) in zip(calibration.targets, raw_points):
            target_norm_x = target_x / screen_w - 0.5
            target_norm_y = target_y / screen_h - 0.5
            # Add a cross-axis term to the samples.  The full affine can model
            # it, but Rake runtime can only use diagonal gain plus offset.
            sample_x = raw_x + raw_y * 0.45
            sample_y = raw_y - raw_x * 0.30
            calibration._gaze_results.append(
                [SimpleNamespace(norm_pog=np.array([sample_x, sample_y])) for _ in range(6)]
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            calibration.SAVE_DIR = Path(tmpdir)
            calibration._fit()
            saved = json.loads((Path(tmpdir) / "last_failed_calibration.json").read_text())

        self.assertFalse(calibration.is_calibrated)
        self.assertIsNone(calibration.correction_values)
        self.assertEqual(len(done), 1)
        self.assertFalse(done[0][0])
        self.assertGreater(done[0][1], saved["max_accepted_affine_error_px"])
        self.assertFalse(saved["accepted"])
        self.assertIn("affine calibration error too high", saved["failure_reason"])
        self.assertIn("diagnostics", saved)
        self.assertIn("correction_values", saved)
        self.assertFalse((Path(tmpdir) / "last_calibration.json").exists())

    def test_fit_uses_point_median_and_rejects_internal_outliers(self):
        screen_w, screen_h = 1000, 800
        done = []
        calibration = EyeCalibration(
            screen_w,
            screen_h,
            num_points=5,
            on_done=lambda success, error: done.append((success, error)),
        )
        tracker = _FakeTracker()
        calibration.start(tracker)

        expected_gain_x = 1.1
        expected_gain_y = 0.9
        expected_bias_x = 0.03
        expected_bias_y = -0.02
        calibration._gaze_results = []
        for target_x, target_y in calibration.targets:
            target_norm_x = target_x / screen_w - 0.5
            target_norm_y = target_y / screen_h - 0.5
            raw_x = (target_norm_x - expected_bias_x) / expected_gain_x
            raw_y = (target_norm_y - expected_bias_y) / expected_gain_y
            good_samples = [
                SimpleNamespace(norm_pog=np.array([raw_x + dx, raw_y + dy]))
                for dx, dy in [
                    (0.000, 0.000),
                    (0.004, -0.002),
                    (-0.003, 0.003),
                    (0.002, 0.001),
                    (-0.002, -0.001),
                ]
            ]
            bad_samples = [
                SimpleNamespace(norm_pog=np.array([raw_x + 0.8, raw_y - 0.8])),
                SimpleNamespace(norm_pog=np.array([np.nan, raw_y])),
            ]
            calibration._gaze_results.append(good_samples + bad_samples)

        with tempfile.TemporaryDirectory() as tmpdir:
            calibration.SAVE_DIR = Path(tmpdir)
            calibration._fit()
            saved = json.loads((Path(tmpdir) / "last_calibration.json").read_text())

        self.assertTrue(calibration.is_calibrated)
        self.assertTrue(done[0][0])
        affine = np.array(calibration.correction_values["rake_affine_matrix"], dtype=np.float64)
        self.assertAlmostEqual(affine[0, 0], expected_gain_x, places=2)
        self.assertAlmostEqual(affine[1, 1], expected_gain_y, places=2)
        for point in saved["diagnostics"]["points"]:
            self.assertEqual(point["sample_count_raw"], 7)
            self.assertEqual(point["sample_count_valid"], 6)
            self.assertLess(point["sample_count_kept"], point["sample_count_valid"])
            self.assertIn("raw_median_norm", point)

    def test_fit_rejects_single_extreme_point_without_saving_as_success(self):
        screen_w, screen_h = 1000, 800
        done = []
        calibration = EyeCalibration(
            screen_w,
            screen_h,
            num_points=5,
            on_done=lambda success, error: done.append((success, error)),
        )
        tracker = _FakeTracker()
        calibration.start(tracker)

        calibration._gaze_results = []
        for point_idx, (target_x, target_y) in enumerate(calibration.targets):
            target_norm_x = target_x / screen_w - 0.5
            target_norm_y = target_y / screen_h - 0.5
            raw_x = target_norm_x
            raw_y = target_norm_y
            if point_idx == 4:
                raw_x += 1.0
                raw_y += 1.0
            calibration._gaze_results.append(
                [SimpleNamespace(norm_pog=np.array([raw_x, raw_y])) for _ in range(5)]
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            calibration.SAVE_DIR = Path(tmpdir)
            calibration._fit()
            failed_path = Path(tmpdir) / "last_failed_calibration.json"
            saved = json.loads(failed_path.read_text())

        self.assertFalse(calibration.is_calibrated)
        self.assertFalse(done[0][0])
        self.assertFalse((Path(tmpdir) / "last_calibration.json").exists())
        self.assertFalse(saved["accepted"])
        self.assertIn("one calibration point error too high", saved["failure_reason"])

    def test_fit_uses_reinferred_pog_after_adaptation(self):
        """After adaptation, the affine fit must use fresh model output, not
        the stale pre-adaptation norm_pog collected during calibration."""
        screen_w, screen_h = 1000, 800
        done = []
        calibration = EyeCalibration(
            screen_w,
            screen_h,
            num_points=5,
            on_done=lambda success, error: done.append((success, error)),
        )
        class _AdaptingTracker(_FakeTracker):
            def __init__(self, targets_norm):
                super().__init__()
                self.infer_calls = 0
                self._targets_norm = targets_norm

            def infer_fn(self, *, image, head_vector, face_origin_3d):
                self.infer_calls += 1
                # The point index is smuggled through head_vector so this
                # fake can return that point's exact target -- proving the
                # affine fit picks up re-inferred values, not the bogus
                # norm_pog=(5, 5) stored on every sample below.
                point_idx = int(np.asarray(head_vector)[0][0])
                return np.array([self._targets_norm[point_idx]], dtype=np.float64)

        tracker = _AdaptingTracker([])
        calibration.start(tracker)
        tracker._targets_norm = [
            (tx / screen_w - 0.5, ty / screen_h - 0.5) for tx, ty in calibration.targets
        ]

        calibration._gaze_results = []
        for point_idx, _ in enumerate(calibration.targets):
            calibration._gaze_results.append(
                [
                    SimpleNamespace(
                        norm_pog=np.array([5.0, 5.0]),
                        eye_patch=np.zeros((8, 8, 3), dtype=np.uint8),
                        head_vector=np.array([point_idx, 0, 0], dtype=np.float32),
                        face_origin_3d=np.zeros(3, dtype=np.float32),
                    )
                    for _ in range(5)
                ]
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            calibration.SAVE_DIR = Path(tmpdir)
            calibration._fit()

        self.assertTrue(tracker.adapt_called)
        self.assertGreater(tracker.infer_calls, 0)
        self.assertTrue(calibration.is_calibrated)
        self.assertLess(done[0][1], 1e-6)
        affine = np.array(calibration.correction_values["rake_affine_matrix"], dtype=np.float64)
        # Re-inferred points map exactly onto their targets, so the fitted
        # affine should be close to identity, not skewed by norm_pog=(5, 5).
        self.assertAlmostEqual(affine[0, 0], 1.0, places=3)
        self.assertAlmostEqual(affine[1, 1], 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
