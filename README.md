# 智能经营分析系统（Econanical Intelligence System）

智能经营分析系统是一套面向企业财务团队的数据分析平台，支持对月度经营数据的上传、预览、建模预测与报告导出，帮助管理者快速洞察企业经营走势并输出专业化决策建议。

## 核心功能
- **数据上传与预览**：通过单一入口上传月度财务数据，完成格式校验、字段预览和列映射配置。
- **自动建模预测**：结合时间序列与机器学习模型，对营收、利润等关键指标生成未来 12 个月预测，并计算置信区间。
- **财务健康度评估**：依据流动比率、资产负债率、净利润率等指标计算企业健康度评分，并提供建议。
- **任务管理与可视化仪表盘**：异步任务执行，实时查看状态，预测结果以图表方式呈现。
- **多格式导出**：支持导出 Excel、CSV 与 PDF 报告，满足内部汇报与归档需求。

## 系统架构
- `app.py`：Flask 应用入口，负责初始化核心算法、任务管理器与路由。
- `model/core_algorithm.py`：核心算法模块，提供财务预测、健康度评估及资金优化等能力。
- `model/model_training.py`：训练与预测管线，实现特征工程、模型训练及结果解释。
- `model/routes.py`：HTTP 路由定义，封装文件上传、任务启动、结果查询、数据导出等 API。
- `model/task_manager.py`：任务生命周期与日志管理，支持异步执行。
- `templates/` 与 `static/`：前端页面、样式与交互脚本，构建上传流程与仪表盘。

## 快速开始
1. **环境准备**
   - Python 3.9+
   - 建议在虚拟环境中安装依赖：`python -m venv .venv && source .venv/bin/activate`（Windows 使用 `.\.venv\Scripts\activate`）
2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```
3. **启动服务**
   ```bash
   python app.py
   ```
   默认监听 `http://localhost:5000`。

## 目录结构
```
project_folder/
├─ app.py
├─ config.yaml
├─ logging.yaml
├─ model/
│  ├─ core_algorithm.py
│  ├─ model_training.py
│  ├─ report_generator.py
│  ├─ routes.py
│  └─ task_manager.py
├─ templates/
│  ├─ index.html
│  ├─ dashboard.html
│  └─ base.html
├─ static/
│  ├─ css/
│  ├─ js/
│  └─ img/
└─ Uploads/   # 用户上传文件的临时存储目录
```

## 主要 API 概览
| 方法 | 路径 | 描述 |
| ---- | ---- | ---- |
| `POST` | `/upload_preview` | 上传月度数据文件并返回列信息及数据预览 |
| `POST` | `/start_task` | 启动训练与预测任务，返回任务 ID |
| `GET` | `/task_status` | 查询任务执行状态与进度 |
| `GET` | `/get_result` | 获取任务结果及仪表盘链接 |
| `GET` | `/export_data` | 导出预测数据（支持 `xlsx`/`csv`） |
| `GET` | `/export_report` | 导出 PDF 财务预测报告 |

更多接口细节可参阅 `swagger.yaml` 或 `DEPLOYMENT.md`。

## 任务流程
1. 上传月度财务数据并确认预览。
2. 根据需要调整列映射并启动任务。
3. 等待任务完成后查看仪表盘、健康度分析与建议。
4. 按需导出预测数据或 PDF 报告。

## 常见问题
- **Prophet / PyMC 缺失**：安装依赖前请确保已配置对应的系统环境（如 `cmdstanpy` 编译器）。
- **字体缺失导致 PDF 乱码**：在 Windows 上请确认 `C:/Windows/Fonts/simhei.ttf` 或其他中文字体可用。
- **上传文件未清理**：系统会定时清理 `Uploads/` 目录，可在 `model/file_utils.py` 中调整策略。

如需进一步扩展或集成，请参考代码注释与模块设计，或联系项目维护者。欢迎提交 Issues 与 Pull Requests，共同完善系统。***