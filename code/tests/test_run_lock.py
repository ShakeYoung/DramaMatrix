"""R1 项目运行锁测试：同一项目禁止并发生产进程。"""

import os
import tempfile
import unittest

import src.db as db_module
from src.run_lock import ProjectLockHeldError, acquire_project_lock


class RunLockTests(unittest.TestCase):
    def setUp(self):
        self._original = db_module.DB_PATH
        self.tmp = tempfile.TemporaryDirectory()
        db_module.DB_PATH = os.path.join(self.tmp.name, "lock.db")

    def tearDown(self):
        db_module.DB_PATH = self._original
        self.tmp.cleanup()

    def test_second_acquire_for_same_project_raises(self):
        fd1, path1 = acquire_project_lock("proj_a")
        self.addCleanup(os.close, fd1)
        with self.assertRaises(ProjectLockHeldError) as ctx:
            acquire_project_lock("proj_a")
        self.assertIn("proj_a", str(ctx.exception))
        self.assertIn("pid=", ctx.exception.holder, "应报告持有方信息便于排查")

    def test_release_allows_reacquire(self):
        fd1, _ = acquire_project_lock("proj_b")
        os.close(fd1)  # fd 关闭即释放
        fd2, _ = acquire_project_lock("proj_b")
        os.close(fd2)

    def test_different_projects_do_not_interfere(self):
        fd1, _ = acquire_project_lock("proj_c")
        self.addCleanup(os.close, fd1)
        fd2, _ = acquire_project_lock("proj_d")
        os.close(fd2)

    def test_lock_file_lives_next_to_db(self):
        fd, path = acquire_project_lock("proj_e")
        self.addCleanup(os.close, fd)
        self.assertEqual(str(path), f"{db_module.DB_PATH}.proj_e.lock")


if __name__ == "__main__":
    unittest.main()
