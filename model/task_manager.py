# task_manager.py
import os
import time
import threading
from typing import Dict, Any, Optional

# 获取项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, 'Logs')


class TaskManager:
    def __init__(self, log_dir: str = LOG_DIR):
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.tasks_lock = threading.Lock()
        self.log_dir = log_dir
        self.cleanup_thread = None

        # 确保日志目录存在
        os.makedirs(self.log_dir, exist_ok=True)

    def update_task_status(self, task_id: str, status: Optional[str] = None,
                           progress: Optional[float] = None,
                           result: Optional[Dict] = None,
                           error: Optional[str] = None) -> None:
        """更新任务状态（线程安全）"""
        with self.tasks_lock:
            if task_id not in self.tasks:
                self.tasks[task_id] = {
                    'status': 'running',
                    'progress': 0,
                    'created_at': time.time(),
                    'result': None,
                    'error': None
                }

            if status is not None:
                self.tasks[task_id]['status'] = status
            if progress is not None:
                self.tasks[task_id]['progress'] = progress
            if result is not None:
                self.tasks[task_id]['result'] = result
            if error is not None:
                self.tasks[task_id]['error'] = error

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """获取任务状态（线程安全）"""
        with self.tasks_lock:
            return self.tasks.get(task_id, {}).copy()

    def log_message(self, task_id: str, message: str) -> None:
        """记录任务日志"""
        log_file = os.path.join(self.log_dir, f'task_{task_id}.log')
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")

    def _cleanup_old_tasks(self, max_age_hours: int = 24) -> None:
        """清理超过指定时间的任务（内部方法）"""
        while True:
            try:
                current_time = time.time()
                with self.tasks_lock:
                    to_delete = []
                    for task_id, task_info in self.tasks.items():
                        if current_time - task_info.get('created_at', 0) > max_age_hours * 3600:
                            to_delete.append(task_id)

                    for task_id in to_delete:
                        # 删除相关日志文件
                        log_file = os.path.join(self.log_dir, f'task_{task_id}.log')
                        if os.path.exists(log_file):
                            try:
                                os.remove(log_file)
                            except:
                                pass
                        del self.tasks[task_id]

                # 每小时清理一次
                time.sleep(3600)
            except Exception as e:
                print(f"清理任务时出错: {e}")
                time.sleep(300)  # 出错后等待5分钟再试

    def start_cleanup_thread(self, max_age_hours: int = 24) -> None:
        """启动清理线程"""
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_old_tasks,
            args=(max_age_hours,),
            daemon=True
        )
        self.cleanup_thread.start()