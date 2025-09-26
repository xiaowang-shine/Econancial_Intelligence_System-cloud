\# 给比赛主办方的部署说明



\## 系统技术栈

\- \*\*后端\*\*: Flask + Python 3.9

\- \*\*机器学习\*\*: pymc3, Prophet, scikit-learn

\- \*\*前端\*\*: HTML/CSS/JavaScript + Chart.js



\## 部署要求

\- \*\*内存\*\*: 4GB+ (推荐8GB，因包含贝叶斯模型)

\- \*\*存储\*\*: 2GB可用空间

\- \*\*网络\*\*: 可访问外部Python包仓库



\## 一键部署方案



\### 方案A: GitHub Codespaces (推荐)

✅ 完全免费  

✅ 无需服务器配置  

✅ 5分钟即可运行  

✅ 内置版本控制



访问: \[GitHub仓库] -> Code -> Codespaces -> 创建新Codespace



\### 方案B: 传统服务器部署

如需在自有服务器部署，请运行：

```bash

git clone \[仓库地址]

cd Econanical\_Intelligence\_System

pip install -r requirements.txt

python app.py

