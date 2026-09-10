"""R2 交付包验证测试：完整性门禁阻断 growth_ready，哈希入清单。"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.state import (
    EpisodeScriptData,
    EpisodeState,
    FeedbackLog,
    GrowthAsset,
    GrowthMeta,
    ShotStoryboard,
)


def make_episode(status="edit_completed"):
    return EpisodeState(
        status=status,
        script_data=EpisodeScriptData(ep_id="ep_01", outline="复仇", ending_hook="摔杯"),
        storyboard_data=[
            ShotStoryboard(shot_id="s01", camera="Static", visual_prompt="v",
                           dialogue="你以为你赢定了？", speaker="女主", duration="4s", audio="tense"),
        ],
        growth_meta=GrowthMeta(title="测试成片", description="desc", tags=["女频"], cover_prompt="cover"),
    )


def write_clips(directory: Path):
    clips = []
    for name in ("ep_01_hook", "ep_01_climax"):
        path = directory / f"{name}.mp4"
        path.write_bytes(b"clip-" + name.encode())
        clips.append(GrowthAsset(name=name.rsplit("_", 1)[1], path=str(path),
                                 start_seconds=0.0, duration_seconds=8.0,
                                 headline="h", description="d", tags=["t"]))
    return clips


class PublishPackageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.ep = make_episode()
        self.ep.growth_assets = write_clips(self.base)
        self.ep.final_video_path = str(self.base / "master.mp4")
        (self.base / "master.mp4").write_bytes(b"master")

    def test_export_records_sha256_and_verifies(self):
        from src.publish import export_publish_package, verify_publish_package

        pkg = export_publish_package("dgv_p", "ep_01", self.ep, output_dir=self.base / "pkg")
        self.assertIsNotNone(pkg)
        meta = json.loads((pkg / "publish_meta.json").read_text(encoding="utf-8"))
        for clip in meta["clips"]:
            self.assertTrue(clip.get("sha256"), "切片必须带 SHA-256")
        self.assertEqual(len(meta["clips"]), 2)
        self.assertEqual(verify_publish_package(pkg, self.ep), [], "完整包应通过验证")

    def test_missing_clip_file_blocks_verification(self):
        from src.publish import export_publish_package, verify_publish_package

        # 导出前源切片就被删（磁盘事故/路径漂移）：切片进不了包，验证必须拦截。
        Path(self.ep.growth_assets[0].path).unlink()
        pkg = export_publish_package("dgv_p", "ep_01", self.ep, output_dir=self.base / "pkg")
        problems = verify_publish_package(pkg, self.ep)
        self.assertTrue(any("ep_01_hook" in p for p in problems))

    def test_packaged_clip_deleted_after_export_fails(self):
        from src.publish import export_publish_package, verify_publish_package

        pkg = export_publish_package("dgv_p", "ep_01", self.ep, output_dir=self.base / "pkg")
        (pkg / "ep_01_hook.mp4").unlink()
        problems = verify_publish_package(pkg, self.ep)
        self.assertTrue(any("ep_01_hook" in p for p in problems))

    def test_tampered_package_fails_hash_check(self):
        from src.publish import export_publish_package, verify_publish_package

        pkg = export_publish_package("dgv_p", "ep_01", self.ep, output_dir=self.base / "pkg")
        packaged = pkg / "ep_01_hook.mp4"
        packaged.write_bytes(b"tampered")
        problems = verify_publish_package(pkg, self.ep)
        self.assertTrue(any("哈希" in p for p in problems))

    def test_missing_zip_is_a_problem(self):
        from src.publish import export_publish_package, verify_publish_package

        pkg = export_publish_package("dgv_p", "ep_01", self.ep, output_dir=self.base / "pkg")
        pkg.with_suffix(".zip").unlink()
        problems = verify_publish_package(pkg, self.ep)
        self.assertTrue(any("ZIP" in p for p in problems))


class Agent7DeliveryGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        # 本用例聚焦交付门禁；关闭整集验收并登记清理，避免泄漏到其他用例。
        self._episode_review = os.environ.get("DRAMAMATRIX_EPISODE_REVIEW")
        os.environ["DRAMAMATRIX_EPISODE_REVIEW"] = "0"

        def _restore():
            if self._episode_review is None:
                os.environ.pop("DRAMAMATRIX_EPISODE_REVIEW", None)
            else:
                os.environ["DRAMAMATRIX_EPISODE_REVIEW"] = self._episode_review

        self.addCleanup(_restore)

    def _run_agent7(self, ep):
        from src.agents.agent7_growth import process_agent7_growth

        state = {
            "project_id": "dgv7_p", "meta_info": {"genre_tags": ["女频"]},
            "market_feedback": None, "source_material": {}, "master_script_outline": "",
            "episodes": {"ep_01": ep}, "characters": [], "task_cycle": 1,
            "scout_attempts": 0, "system_status": "x",
        }
        with patch.dict(os.environ, {"DRAMAMATRIX_OUTPUT_DIR": str(self.base)}, clear=False), patch(
            "src.agents.agent7_growth.video_duration", return_value=20.0
        ), patch(
            "src.agents.agent7_growth.cut_video",
            side_effect=lambda src, dst, start, dur: (
                dst.parent.mkdir(parents=True, exist_ok=True), dst.write_bytes(b"clip"), dst
            )[2],
        ):
            process_agent7_growth(state)
        return state

    def test_growth_ready_only_after_verified_package(self):
        ep = make_episode()
        ep.final_video_path = str(self.base / "master.mp4")
        (self.base / "master.mp4").write_bytes(b"master")
        state = self._run_agent7(ep)
        self.assertEqual(ep.status, "growth_ready")
        pkg_meta = Path(ep.deliverables[-1].path)
        self.assertTrue(pkg_meta.is_file())

    def test_export_failure_blocks_growth_ready(self):
        ep = make_episode()
        ep.final_video_path = str(self.base / "master.mp4")
        (self.base / "master.mp4").write_bytes(b"master")
        with patch("src.publish.export_publish_package", side_effect=RuntimeError("disk full")):
            state = self._run_agent7(ep)
        self.assertEqual(ep.status, "growth_failed", "导出异常必须阻断而非只打告警")
        self.assertTrue(any(fb.reason_code == "GROWTH_EXPORT_FAILED" for fb in ep.feedback_log))

    def test_verification_failure_blocks_growth_ready(self):
        ep = make_episode()
        ep.final_video_path = str(self.base / "master.mp4")
        (self.base / "master.mp4").write_bytes(b"master")

        # 导出"成功"，但验证发现切片缺失（cut 写出的文件被删一个）。
        original_cut_behavior = lambda src, dst, start, dur: (
            dst.parent.mkdir(parents=True, exist_ok=True), dst.write_bytes(b"clip"), dst
        )[2]

        from src.agents.agent7_growth import process_agent7_growth

        state = {
            "project_id": "dgv7_p", "meta_info": {"genre_tags": ["女频"]},
            "market_feedback": None, "source_material": {}, "master_script_outline": "",
            "episodes": {"ep_01": ep}, "characters": [], "task_cycle": 1,
            "scout_attempts": 0, "system_status": "x",
        }

        def cut_deleting_clips(src, dst, start, dur):
            result = original_cut_behavior(src, dst, start, dur)
            if "hook" in dst.name:
                dst.unlink()  # hook 切片写完即被"丢失"
            return result

        with patch.dict(os.environ, {"DRAMAMATRIX_OUTPUT_DIR": str(self.base)}, clear=False), patch(
            "src.agents.agent7_growth.video_duration", return_value=20.0
        ), patch("src.agents.agent7_growth.cut_video", side_effect=cut_deleting_clips):
            process_agent7_growth(state)
        self.assertEqual(ep.status, "growth_failed", "切片缺失不得标记 growth_ready")
        self.assertTrue(any("投放包完整性验证未通过" in fb.message for fb in ep.feedback_log))


if __name__ == "__main__":
    unittest.main()
