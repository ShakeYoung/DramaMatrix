"""Tests for reproducibility & evaluation infrastructure (V1–V5).

Covers: media integrity + sha256 (V1), scene_seed anti-collision (V1),
GeneratedVideoAsset evidence fields (V1), QC result persistence (V2),
actual-duration-driven timing (V3), run_context secret exclusion + git sha
(V4), state_history append (V4), manual score import/export (V5), reference
asset chain (V5), agnes_usage extended columns (V5).
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.state import EpisodeState, EpisodeScriptData, GeneratedVideoAsset, ShotStoryboard


def make_shot(shot_id, dialogue="", duration="4s", scene_id=None):
    kw = dict(shot_id=shot_id, camera="c", visual_prompt="v", dialogue=dialogue,
              duration=duration, audio="a")
    if scene_id:
        kw["scene_id"] = scene_id
    return ShotStoryboard(**kw)


class MediaIntegrityTests(unittest.TestCase):
    """V1: sha256 + ffprobe-based media integrity."""

    def test_sha256_file_consistent(self):
        from src.agnes_video import sha256_file
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "v.mp4"
            p.write_bytes(b"hello world")
            h1 = sha256_file(p)
            h2 = sha256_file(p)
        self.assertIsNotNone(h1)
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 64)  # SHA-256 hex

    def test_sha256_missing_file_returns_none(self):
        from src.agnes_video import sha256_file
        self.assertIsNone(sha256_file(Path("/nonexistent/file.mp4")))

    def test_media_integrity_degrades_without_ffprobe(self):
        from src.agnes_video import media_integrity
        with tempfile.TemporaryDirectory() as d, patch("src.agnes_video.shutil.which", return_value=None):
            p = Path(d) / "v.mp4"
            p.write_bytes(b"video-bytes")
            info = media_integrity(p)
        self.assertIsNotNone(info["sha256"])
        self.assertEqual(info["file_size_bytes"], len(b"video-bytes"))
        self.assertIsNone(info["actual_duration"])  # degraded


class SceneSeedCollisionTests(unittest.TestCase):
    """V1: FNV-1a scene_seed avoids common collisions."""

    def test_same_scene_same_seed(self):
        from src.continuity import scene_seed
        self.assertEqual(scene_seed(100, "A", 1), scene_seed(100, "A", 2))

    def test_distinct_scenes_distinct_seeds(self):
        from src.continuity import scene_seed
        ids = ["scene_01", "scene_02", "scene_03", "客厅", "卧室", "街道", "屋顶", "书房"]
        seeds = {scene_seed(100, sid, 1) for sid in ids}
        # All distinct scene_ids should yield distinct seeds (no collisions).
        self.assertEqual(len(seeds), len(ids))


class AssetEvidenceTests(unittest.TestCase):
    """V1: GeneratedVideoAsset carries seed/reference/model evidence."""

    def test_asset_new_fields_default_none(self):
        asset = GeneratedVideoAsset(shot_id="s1", video_id="v", status="ok", prompt="p")
        self.assertIsNone(asset.sha256)
        self.assertIsNone(asset.seed)
        self.assertIsNone(asset.model_version)
        self.assertIsNone(asset.reference_image_url)

    def test_asset_round_trips_evidence(self):
        asset = GeneratedVideoAsset(
            shot_id="s1", video_id="v", status="ok", prompt="p",
            sha256="abc123", seed=42, model_version="agnes-video-v2.0",
            actual_duration=4.04, has_audio=True, reference_image_url="data:image/png;base64,xxx",
        )
        data = asset.model_dump(mode="json")
        restored = GeneratedVideoAsset.model_validate(data)
        self.assertEqual(restored.sha256, "abc123")
        self.assertEqual(restored.seed, 42)
        self.assertEqual(restored.actual_duration, 4.04)
        self.assertTrue(restored.has_audio)


class QCPersistenceTests(unittest.TestCase):
    """V2: QC results persist to shot_qc_results."""

    def test_qc_insert_and_query(self):
        import src.db
        with tempfile.TemporaryDirectory() as d:
            src.db.DB_PATH = os.path.join(d, "test.db")
            src.db.init_db()
            src.db.db_insert_qc_result(
                project_id="p1", ep_key="ep_01", shot_id="s1", passed=True,
                brightness_diff=12.3, threshold=25.0, issues=["warn"], metrics={"brightness_diff": 12.3},
            )
            results = src.db.db_query_qc_results("p1", ep_key="ep_01")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["shot_id"], "s1")
        self.assertEqual(results[0]["brightness_diff"], 12.3)
        self.assertIn("brightness_diff", json.loads(results[0]["metrics"]))


class ActualDurationTimingTests(unittest.TestCase):
    """V3: subtitle/TTS timing uses actual_duration when available."""

    def test_aligned_durations_prefers_actual(self):
        from src.agents.agent6_editor import _aligned_durations
        ep = EpisodeState(
            storyboard_data=[make_shot("s1", duration="4s"), make_shot("s2", duration="5s")],
        )
        ep.video_assets = [
            GeneratedVideoAsset(shot_id="s1", video_id="v", status="ok", prompt="p", actual_duration=4.04),
            GeneratedVideoAsset(shot_id="s2", video_id="v", status="ok", prompt="p", actual_duration=5.04),
        ]
        durations = _aligned_durations(ep)
        self.assertEqual(durations, [4.04, 5.04])

    def test_aligned_durations_falls_back_to_planned(self):
        from src.agents.agent6_editor import _aligned_durations
        ep = EpisodeState(storyboard_data=[make_shot("s1", duration="4s")])
        ep.video_assets = []  # no assets → fallback to planned
        self.assertEqual(_aligned_durations(ep), [4.0])


class RunContextTests(unittest.TestCase):
    """V4: run_context excludes secrets and captures git sha."""

    def test_excludes_secrets(self):
        from src.run_context import capture_run_context
        with patch.dict(os.environ, {
            "AGNES_API_KEY": "secret-key",
            "OPENAI_API_KEY": "sk-xxx",
            "DRAMAMATRIX_TTS_PROVIDER": "edge",
        }, clear=False):
            ctx = capture_run_context()
        config = ctx["config"]
        self.assertNotIn("AGNES_API_KEY", config)
        self.assertNotIn("OPENAI_API_KEY", config)
        self.assertEqual(config.get("DRAMAMATRIX_TTS_PROVIDER"), "edge")

    def test_has_git_sha_and_python(self):
        from src.run_context import capture_run_context
        ctx = capture_run_context()
        self.assertIn("git_sha", ctx)
        self.assertIn("python_version", ctx)
        self.assertIn("dependencies", ctx)


class StateHistoryTests(unittest.TestCase):
    """V4: state_history appends (not overwrites)."""

    def test_history_appends_multiple_versions(self):
        import src.db
        with tempfile.TemporaryDirectory() as d:
            src.db.DB_PATH = os.path.join(d, "test.db")
            src.db.init_db()
            state = {"project_id": "p1", "system_status": "s1", "run_context": {"run_id": "r1"}}
            src.db.db_save_project_state(state)
            state["system_status"] = "s2"
            src.db.db_save_project_state(state)
            history = src.db.db_get_state_history("p1")
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["version"], 2)  # latest first
        self.assertEqual(history[1]["version"], 1)


class ManualScoreTests(unittest.TestCase):
    """V5: manual score import/export round-trip."""

    def test_import_export_roundtrip(self):
        import csv
        import src.db
        with tempfile.TemporaryDirectory() as d:
            src.db.DB_PATH = os.path.join(d, "test.db")
            src.db.init_db()
            # Write a CSV
            csv_in = os.path.join(d, "scores.csv")
            with open(csv_in, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["project_id", "ep_key", "shot_id", "character_consistency", "overall"])
                w.writeheader()
                w.writerow({"project_id": "p1", "ep_key": "ep_01", "shot_id": "s1", "character_consistency": "0.8", "overall": "0.75"})
            inserted = src.db.db_import_manual_scores(csv_in)
            self.assertEqual(inserted, 1)
            # Export
            csv_out = os.path.join(d, "export.csv")
            src.db.db_export_manual_scores("p1", csv_out)
            with open(csv_out, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["shot_id"], "s1")
            self.assertEqual(float(rows[0]["character_consistency"]), 0.8)


class ReferenceAssetTests(unittest.TestCase):
    """V5: reference asset chain recorded."""

    def test_insert_reference_asset(self):
        import src.db
        with tempfile.TemporaryDirectory() as d:
            src.db.DB_PATH = os.path.join(d, "test.db")
            src.db.init_db()
            src.db.db_insert_reference_asset(
                project_id="p1", asset_type="tail_frame", ref_id="tail",
                referenced_by_shot="s2", sha256="abc",
            )
        # No exception = pass (insert succeeded)


class AgnesUsageExtendedTests(unittest.TestCase):
    """V5: agnes_usage has extended columns."""

    def test_extended_columns_exist(self):
        import sqlite3
        import src.db
        with tempfile.TemporaryDirectory() as d:
            src.db.DB_PATH = os.path.join(d, "test.db")
            src.db.init_db()
            conn = sqlite3.connect(src.db.DB_PATH)
            cols = {row[1] for row in conn.execute("PRAGMA table_info(agnes_usage)").fetchall()}
            conn.close()
        for col in ("queue_wait_seconds", "render_seconds", "download_seconds", "redraw_count", "cost_estimate", "quality_score"):
            self.assertIn(col, cols)


if __name__ == "__main__":
    unittest.main()