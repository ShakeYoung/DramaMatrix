"""项目运行锁（R1）：同一项目同一时刻只允许一个生产进程。

防止两个进程同时 resume 同一快照、重复提交付费视频任务、互相覆盖保存。
锁为进程持有的 flock：持有进程退出（正常或崩溃）后内核自动释放，
残留的锁文件不阻塞后续获取。review/dashboard 服务只读快照、不写项目状态，
不在加锁范围内。
"""

from __future__ import annotations

import fcntl
import os
import time
from pathlib import Path

import src.db as db_module


class ProjectLockHeldError(RuntimeError):
    """目标项目已被另一个进程持有运行锁。"""

    def __init__(self, project_id: str, holder: str, lock_path: Path):
        self.project_id = project_id
        self.holder = holder
        self.lock_path = lock_path
        super().__init__(
            f"项目 {project_id} 正在被另一个进程运行（持有方：{holder}）。"
            f"如确认无其他进程，可删除锁文件后重试：{lock_path}"
        )


def _lock_path(project_id: str) -> Path:
    # 动态读取 db_module.DB_PATH，与测试替换该属性隔离数据库的做法一致。
    return Path(str(db_module.DB_PATH) + f".{project_id}.lock")


def acquire_project_lock(project_id: str):
    """获取项目独占运行锁；成功返回 (fd, path)，失败抛 ProjectLockHeldError。

    返回的 fd 由调用方持有到进程结束（fd 关闭即释放），无需显式 unlock。
    """
    path = _lock_path(project_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        holder = "未知进程"
        try:
            content = os.pread(fd, 4096, 0).decode("utf-8", errors="replace").strip()
            if content:
                holder = content
        except OSError:
            pass
        os.close(fd)
        raise ProjectLockHeldError(project_id, holder, path) from None
    os.ftruncate(fd, 0)
    os.pwrite(fd, f"pid={os.getpid()} acquired_at={time.strftime('%Y-%m-%d %H:%M:%S')}".encode("utf-8"), 0)
    return fd, path
