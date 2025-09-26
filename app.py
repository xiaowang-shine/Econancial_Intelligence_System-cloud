# app.py —— 智能经营分析系统主入口
import os
import time
from datetime import date
from flask import Flask

# 导入自定义模块
from model.task_manager import TaskManager
from model.file_utils import UPLOAD_DIR, LOG_DIR
from model.core_algorithm import CoreAlgorithm  # 使用新的核心算法
from model.routes import register_routes

# 检测是否是Codespaces环境
IS_CODESPACES = os.getenv('CODESPACES') == 'true'

if IS_CODESPACES:
    # Codespaces特定配置
    HOST = '0.0.0.0'
    PORT = 5000
    DEBUG = False  # 生产环境关闭debug
else:
    # 本地开发配置
    HOST = '0.0.0.0'
    PORT = 5000
    DEBUG = True

# ------------------ 应用初始化 ------------------
app = Flask(__name__, template_folder='templates', static_folder='static')

# 初始化系统组件 - 使用 CoreAlgorithm 替代 system_mod
core_algorithm = CoreAlgorithm(desktop_path="E:/Project1/project_folder/Download")
task_manager = TaskManager(log_dir=LOG_DIR)

# 确保目录存在
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ------------------ 注册路由 ------------------
# 修复：参数顺序正确传递
register_routes(app, core_algorithm, task_manager)  # 注意参数顺序：app, system, task_manager

# ------------------ 启动清理线程 ------------------
task_manager.start_cleanup_thread()

# ------------------ 主程序入口 ------------------
if __name__ == '__main__':
    print("启动智能经营分析系统...")
    if IS_CODESPACES:
        print("运行在GitHub Codespaces环境")
        print(
            f"应用地址: https://{os.getenv('CODESPACE_NAME')}-5000.{os.getenv('GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN')}")
    else:
        print("运行在本地环境")
        print("系统已就绪，访问 http://localhost:5000")

    app.run(debug=DEBUG, host=HOST, port=PORT)





