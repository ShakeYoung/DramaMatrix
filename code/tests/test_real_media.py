"""真实媒体集成测试（W6）。

与 mock 测试互补：用 ffmpeg lavfi 生成真实 MP4/PNG，覆盖抽帧、完整性探测、
拼接、切片、裁剪、QC 真帧比较、审阅缩略图等媒体链路。环境无 ffmpeg 时整文件
跳过（与 CI 之外的开发机行为一致）；CI 已安装 ffmpeg，这些测试真实执行。
"""

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


def _make_clip(path: Path, source: str = "testsrc=s=160x288:d=1", seconds: float = 1.0) -> Path:
    subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", source,
         "-t", f"{seconds:.2f}", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    return path


@unittest.skipUnless(FFMPEG and FFPROBE, "ffmpeg/ffprobe required")
class RealMediaTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self._saved_output = os.environ.get("DRAMAMATRIX_OUTPUT_DIR")
        os.environ["DRAMAMATRIX_OUTPUT_DIR"] = str(self.tmp)
        self.addCleanup(self._restore)

    def _restore(self):
        if self._saved_output is None:
            os.environ.pop("DRAMAMATRIX_OUTPUT_DIR", None)
        else:
            os.environ["DRAMAMATRIX_OUTPUT_DIR"] = self._saved_output

    def test_extract_frames_and_qc_similarity(self):
        from src.agnes_video import extract_first_frame, extract_last_frame
        from src.continuity_qc import frame_similarity

        clip = _make_clip(self.tmp / "shot.mp4")
        head = extract_first_frame(clip, self.tmp / "head.png")
        tail = extract_last_frame(clip, self.tmp / "tail.png")
        self.assertTrue(head and head.is_file())
        self.assertTrue(tail and tail.is_file())
        # 同一素材自比较相似度为 1；与另一构图（纯色）比较低于 1。
        self.assertAlmostEqual(frame_similarity(head, head), 1.0)
        flat = self.tmp / "flat.png"
        subprocess.run(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-f", "lavfi",
             "-i", "color=gray:s=64x64", "-frames:v", "1", str(flat)],
            check=True,
        )
        self.assertLess(frame_similarity(head, flat), 1.0)

    def test_media_integrity_on_real_mp4_and_corrupt_file(self):
        from src.agnes_video import media_integrity

        clip = _make_clip(self.tmp / "ok.mp4")
        info = media_integrity(clip)
        self.assertTrue(info["probe_ok"])
        self.assertEqual((info["width"], info["height"]), (160, 288))
        self.assertIsNotNone(info["actual_duration"])
        self.assertIsNotNone(info["sha256"])

        corrupt = self.tmp / "corrupt.mp4"
        corrupt.write_bytes(b"this is not an mp4")
        bad = media_integrity(corrupt)
        self.assertFalse(bad["probe_ok"])  # P0-3 硬门禁：损坏文件必须被识别

    def test_concat_and_cut_real_clips(self):
        from src.agnes_video import concat_videos, cut_video, video_duration

        first = _make_clip(self.tmp / "a.mp4", seconds=1.0)
        second = _make_clip(self.tmp / "b.mp4", source="testsrc2=s=160x288:d=1", seconds=1.0)
        merged = concat_videos([first, second], self.tmp / "merged.mp4")
        duration = video_duration(merged)
        self.assertGreaterEqual(duration, 1.8)

        clip = cut_video(merged, self.tmp / "clip.mp4", 0.0, 0.5)
        self.assertLessEqual(video_duration(clip), 0.7)

    def test_trim_unstable_frames(self):
        from src.agnes_video import trim_unstable_frames, video_duration

        source = _make_clip(self.tmp / "raw.mp4", seconds=2.0)
        trimmed = trim_unstable_frames(source, self.tmp, head=6, tail=6)
        self.assertLess(video_duration(trimmed), video_duration(source))

    def test_review_manifest_thumbnail_roundtrip(self):
        from src.agents.agent5_director import build_agnes_prompt
        from src.state import EpisodeScriptData, EpisodeState, GeneratedVideoAsset, ShotStoryboard
        from src.review import build_review_manifest

        clip = _make_clip(self.tmp / "s01.mp4")
        from src.agnes_video import extract_first_frame

        head = extract_first_frame(clip, self.tmp / "s01_head.png")
        shot = ShotStoryboard(
            shot_id="s01", camera="Static", visual_prompt="A woman in the rain.",
            dialogue="", duration="5s", audio="rain",
        )
        ep = EpisodeState(
            status="video_generated",
            script_data=EpisodeScriptData(ep_id="ep_01", outline="o", ending_hook="h"),
            storyboard_data=[shot],
            video_assets=[GeneratedVideoAsset(
                shot_id="s01", video_id="v", task_id="t", status="completed",
                prompt=build_agnes_prompt(shot), local_path=str(clip),
            )],
        )
        manifest = build_review_manifest("real_media_project", "ep_01", ep)
        # 缩略图路径与磁盘帧对齐（审阅台据此渲染缩略图墙）。
        self.assertEqual(Path(manifest["shots"][0]["thumbnail"]), head)

    def test_default_checker_real_frames_pass(self):
        from src.continuity_qc import DefaultChecker
        from src.state import ShotStoryboard

        clip = _make_clip(self.tmp / "shot.mp4")
        from src.agnes_video import extract_first_frame, extract_last_frame

        head = extract_first_frame(clip, self.tmp / "h.png")
        tail = extract_last_frame(clip, self.tmp / "t.png")
        shot = ShotStoryboard(
            shot_id="s01", camera="Static", visual_prompt="x", dialogue="",
            duration="5s", audio="a",
        )
        result = DefaultChecker().check(None, tail, clip, shot, head)
        self.assertTrue(result.passed)
        self.assertIn("frame_similarity", result.metrics)
        self.assertIn("brightness_diff", result.metrics)


if __name__ == "__main__":
    unittest.main()
