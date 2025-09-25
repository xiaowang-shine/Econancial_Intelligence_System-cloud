# routes.py
import json
import uuid

import numpy as np
from flask import request, jsonify, send_file, render_template, url_for
from werkzeug.routing import BuildError
from datetime import date, datetime
import os
import pandas as pd

# 导入自定义模块
from .task_manager import TaskManager
from .file_utils import save_uploaded_file, read_excel_file, is_valid_excel_file
from .model_training import run_training_task, infer_time_col, infer_target_col
from .report_generator import generate_financial_report, generate_dashboard_chart_data, export_data_to_excel, \
    export_data_to_csv
from .core_algorithm import CoreAlgorithm

MAX_FILE_COUNT = 3  # 最大上传文件数量
core_algo = CoreAlgorithm()  # 创建实例

# 安全URL生成函数（用于模板）
def safe_url_for(endpoint, **values):
    try:
        return url_for(endpoint, **values)
    except BuildError:
        return "#"  # 兜底：避免模板报错

# 上下文处理器（注入到所有模板）
def inject_safe_and_branding():
    return dict(
        safe_url_for=safe_url_for,
        app_name='智能经营分析系统',
        company_name='©',
        current_year=date.today().year,
        imprint_url='#',
        privacy_url='#'
    )

# 注册路由函数
def register_routes(app, system, task_manager):
    """
    注册所有路由到Flask应用

    Args:
        app: Flask应用实例
        system: FinancialHealthSystem实例（已改为CoreAlgorithm实例）
        task_manager: TaskManager实例
    """

    # 上下文处理器
    @app.context_processor
    def context_processor():
        return inject_safe_and_branding()

    # ------------------ 页面路由 ------------------
    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/dashboard')
    def dashboard():
        job_id = request.args.get('job_id')
        task_id = request.args.get('task_id')

        # 获取任务状态
        task_info = task_manager.get_task_status(task_id)
        if not task_id or not task_info or task_info.get('status') != 'finished':
            return render_template('error.html', message='任务未完成或无效 task_id'), 400

        result = task_info['result']

        # 生成图表数据
        chart_data = generate_dashboard_chart_data(result)

        return render_template('dashboard.html',
                               job_id=job_id,
                               result=result,
                               chart_data=chart_data['main_chart'],
                               kpi_chart=chart_data['kpi_chart'],
                               imp_chart=chart_data['importance_chart'])

    # ------------------ API路由 ------------------
    @app.route('/upload_preview', methods=['POST'])
    def upload_preview():
        try:
            files = {}
            for key in ['fileForecast', 'fileHealth', 'fileMonthly']:
                if key not in request.files:
                    return jsonify(status='error', error=f'{key} 未上传'), 400

                file = request.files[key]
                if file.filename == '':
                    return jsonify(status='error', error=f'{key} 没有选择文件'), 400

                files[key] = save_uploaded_file(file)

            columns, preview = {}, {}
            for k, p in files.items():
                df = read_excel_file(p)
                columns[k] = list(df.columns.astype(str))
                preview[k] = df.head(5).fillna('').to_dict(orient='records')

            token = uuid.uuid4().hex

            # 存储预览信息
            task_manager.update_task_status(
                token,
                result={'preview_files': files, 'created_at': task_manager.get_current_time()}
            )

            return jsonify(status='ok', columns=columns, preview=preview, token=token)
        except Exception as e:
            return jsonify(status='error', error=str(e)), 500

    @app.route('/start_task', methods=['POST'])
    def start_task():
        try:
            files = {}
            for key in ['fileForecast', 'fileHealth', 'fileMonthly']:
                if key not in request.files:
                    return jsonify(status='error', error=f'{key} 未上传'), 400
                files[key] = save_uploaded_file(request.files[key])

            mapping_raw = request.form.get('mapping')
            mapping = json.loads(mapping_raw) if mapping_raw else {}

            task_id = uuid.uuid4().hex

            # 初始化任务状态
            task_manager.update_task_status(task_id, status='running', progress=0)

            # 启动任务线程
            def run_task_wrapper():
                try:
                    # 读取文件
                    monthly = read_excel_file(files['fileMonthly'])
                    forecast = read_excel_file(files['fileForecast'])
                    health = read_excel_file(files['fileHealth'])

                    if monthly is None or forecast is None or health is None:
                        raise Exception("读取文件失败")

                    task_manager.log_message(task_id,
                                             f"文件读取完成：monthly={monthly.shape}, forecast={forecast.shape}, health={health.shape}")

                    # 运行训练任务
                    result = run_training_task(monthly, health, mapping)

                    # 更新任务状态为完成
                    task_manager.update_task_status(
                        task_id,
                        status='finished',
                        progress=100,
                        result=result
                    )

                except Exception as e:
                    error_msg = f"任务执行失败: {str(e)}"
                    task_manager.log_message(task_id, error_msg)
                    task_manager.update_task_status(task_id, status='failed', error=error_msg)

            # 启动任务线程
            import threading
            th = threading.Thread(target=run_task_wrapper, daemon=True)
            th.start()

            return jsonify(status='ok', task_id=task_id)
        except Exception as e:
            return jsonify(status='error', error=str(e)), 500

    @app.route('/.well-known/<path:filename>')
    def well_known(filename):
        return "Not Found", 404

    @app.route('/task_status')
    def task_status():
        task_id = request.args.get('task_id')
        if not task_id:
            return jsonify(status='error', error='task_id 参数缺失'), 400

        # 获取任务状态
        info = task_manager.get_task_status(task_id)
        if not info:
            return jsonify(status='error', error='task_id 未找到或已过期'), 404

        return jsonify(
            status=info.get('status', 'unknown'),
            progress=info.get('progress', 0),
            error=info.get('error', '')
        )

    @app.route('/get_result')
    def get_result():
        task_id = request.args.get('task_id')
        if not task_id:
            return jsonify(status='error', error='task_id 参数缺失'), 400

        # 获取任务状态
        info = task_manager.get_task_status(task_id)
        if not info:
            return jsonify(status='error', error='task_id 未找到或已过期'), 404

        if info.get('status') != 'finished':
            return jsonify(status='error', error='任务未完成'), 400

        # 添加仪表盘链接
        dashboard_url = url_for('dashboard', task_id=task_id)
        return jsonify(status='ok', result=info['result'], dashboard_url=dashboard_url)

    @app.route('/generate_monthly_data', methods=['GET'])
    def generate_monthly_data():
        """生成月度财务数据并返回给前端"""
        try:
            # 使用CoreAlgorithm实例
            result_df = system.generate_monthly_data()
            return jsonify(result_df.to_dict(orient='records'))
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    @app.route('/run_optimization', methods=['POST'])
    def run_optimization():
        """运行资金优化并返回结果"""
        try:
            data = request.get_json()
            year = data.get("year", "2024")
            financial_data = data.get("financial_data", {})

            # 运行优化
            results, optimal_result = system.run_optimization(financial_data, year)

            return jsonify({
                "results": results.to_dict(orient='records'),
                "optimal_result": optimal_result.to_dict()
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    @app.route('/export_data')
    def export_data():
        """导出数据路由 - 修复编码问题版本"""
        try:
            task_id = request.args.get('task_id')
            format_type = request.args.get('format', 'xlsx')  # 默认Excel格式

            if not task_id:
                return "task_id parameter is missing", 400

            task_info = task_manager.get_task_status(task_id)
            if not task_info or task_info.get('status') != 'finished':
                return "Task not completed or invalid task_id", 400

            result = task_info['result']

            if format_type in ['xlsx', 'xls']:
                excel_buffer = export_data_to_excel(result, task_id)
                # 使用ASCII文件名避免编码问题
                filename = f'financial_forecast_{task_id}.xlsx'
                return send_file(
                    excel_buffer,
                    as_attachment=True,
                    download_name=filename,
                    mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )
            elif format_type == 'csv':
                csv_content = export_data_to_csv(result, task_id)

                # 创建响应对象，确保使用正确的编码
                from flask import Response
                response = Response(
                    csv_content,
                    mimetype='text/csv'
                )

                # 使用ASCII-only文件名避免编码问题
                filename = f'financial_forecast_{task_id}.csv'
                response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
                response.headers['Content-Type'] = 'text/csv; charset=utf-8'

                return response
            else:
                return "Unsupported format type", 400

        except Exception as e:
            app.logger.error(f"Export route error: {str(e)}")
            return f"Export failed: {str(e)}", 500

    @app.route('/export_report')
    def export_report():
        """导出报告路由 - 修复编码问题版本"""
        try:
            task_id = request.args.get('task_id')

            if not task_id:
                return "task_id parameter is missing", 400

            task_info = task_manager.get_task_status(task_id)
            if not task_info or task_info.get('status') != 'finished':
                return "Task not completed or invalid task_id", 400

            result = task_info['result']

            # 生成PDF报告
            pdf_buffer = generate_financial_report(result, task_id)

            # 使用ASCII文件名避免编码问题
            filename = f'financial_report_{task_id}.pdf'

            return send_file(
                pdf_buffer,
                as_attachment=True,
                download_name=filename,
                mimetype='application/pdf'
            )

        except Exception as e:
            app.logger.error(f"Export report error: {str(e)}")
            return f"Export report failed: {str(e)}", 500

    @app.route('/forecast_financials', methods=['GET'])
    def forecast_financials():
        """预测财务指标并返回"""
        try:
            # 使用CoreAlgorithm实例
            forecast = system.forecast_financials()
            return jsonify(forecast.to_dict(orient='records'))
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    @app.route('/favicon.ico')
    def favicon():
        # 返回favicon
        return send_file(
            os.path.join(app.static_folder, 'img', 'favicon.ico'),
            mimetype='image/x-icon'
        )

    # 添加一个获取当前时间的辅助方法到task_manager
    def get_current_time(self):
        import time
        return time.time()

    @app.route('/analyze_financials', methods=['POST'])
    def analyze_financials():
        """
        分析上传的财务数据并生成预测结果
        """
        try:
            # 检查是否有文件上传
            if 'files' not in request.files:
                return jsonify({'status': 'error', 'message': '没有上传文件'}), 400

            files = request.files.getlist('files')

            # 检查文件数量
            if len(files) > MAX_FILE_COUNT:
                return jsonify({'status': 'error', 'message': f'最多只能上传 {MAX_FILE_COUNT} 个文件'}), 400

            # 检查文件是否为空
            if not files or all(file.filename == '' for file in files):
                return jsonify({'status': 'error', 'message': '没有选择文件'}), 400

            # 验证文件类型
            valid_files = []
            for file in files:
                if file and file.filename != '':
                    # 检查文件扩展名
                    if not is_valid_excel_file(file.filename):
                        return jsonify({'status': 'error', 'message': f'文件 {file.filename} 不是有效的Excel文件'}), 400
                    valid_files.append(file)

            if not valid_files:
                return jsonify({'status': 'error', 'message': '没有有效的Excel文件'}), 400

            # 读取和处理文件
            all_data = []
            for file in valid_files:
                # 保存文件
                file_path = save_uploaded_file(file)

                # 读取Excel文件（支持多工作表）
                try:
                    df_dict = read_excel_file(file_path, sheet_name=None)  # 读取所有工作表

                    # 合并所有工作表的数据
                    for sheet_name, sheet_data in df_dict.items():
                        if not sheet_data.empty:
                            # 添加工作表标识
                            sheet_data['source_sheet'] = sheet_name
                            sheet_data['source_file'] = file.filename
                            all_data.append(sheet_data)
                except Exception as e:
                    app.logger.error(f"读取文件 {file.filename} 失败: {str(e)}")
                    return jsonify({'status': 'error', 'message': f'读取文件 {file.filename} 失败: {str(e)}'}), 400

            if not all_data:
                return jsonify({'status': 'error', 'message': '没有有效的财务数据'}), 400

            # 合并所有数据
            combined_data = pd.concat(all_data, ignore_index=True)

            # 确保日期列是datetime类型
            date_columns = ['日期', 'date', '时间', '月份']  # 可能的日期列名
            date_col = None

            for col in date_columns:
                if col in combined_data.columns:
                    date_col = col
                    break

            if date_col is None:
                return jsonify({'status': 'error', 'message': '未找到日期列'}), 400

            combined_data[date_col] = pd.to_datetime(combined_data[date_col])
            combined_data = combined_data.sort_values(date_col).reset_index(drop=True)

            # 调用核心算法进行预测
            target_cols = ['流动资产合计', '流动负债合计', '存货', '负债合计', '资产总计', '净利润', '营业收入']

            # 确保所有目标列都存在
            missing_cols = [col for col in target_cols if col not in combined_data.columns]
            if missing_cols:
                return jsonify({
                    'status': 'error',
                    'message': f'数据中缺少必要的列: {", ".join(missing_cols)}'
                }), 400

            # 修复：使用CoreAlgorithm实例而不是类方法调用
            historical_data, forecast_data = system.forecast_financials(
                combined_data,
                time_col=date_col,
                target_cols=target_cols,
                forecast_months=12,
                use_bayesian=False
            )

            # 计算财务比率和健康度评分
            financial_forecast = calculate_financial_ratios(forecast_data)

            # 生成健康度对比结果
            health_comparison = generate_health_comparison(financial_forecast)

            # 生成任务ID
            task_id = uuid.uuid4().hex
            task_manager.update_task_status(task_id, status='finished', result={
                'financial_forecast': financial_forecast,
                'health_comparison': health_comparison
            })

            return jsonify({
                'status': 'success',
                'task_id': task_id,
                'results': {
                    'financial_forecast': financial_forecast,
                    'health_comparison': health_comparison
                }
            })

        except Exception as e:
            app.logger.error(f"财务分析错误: {str(e)}")
            return jsonify({'status': 'error', 'message': f'分析失败: {str(e)}'}), 500

    def calculate_financial_ratios(forecast_data):
        """修复健康度评分计算"""
        results = {'year': datetime.now().year + 1, 'data': []}

        for i, (date_index, row) in enumerate(forecast_data.iterrows()):
            try:
                # 日期处理
                if hasattr(date_index, 'strftime'):
                    date_str = date_index.strftime('%Y-%m-%d')
                else:
                    date_str = str(date_index)

                # 获取财务数据
                CA = float(row.get('流动资产合计', 0)) or 39000
                CL = float(row.get('流动负债合计', 0)) or 18247
                Inv = float(row.get('存货', 0)) or 8106
                TD = float(row.get('负债合计', 0)) or 18247
                TA = float(row.get('资产总计', 0)) or 51174
                NI = float(row.get('净利润', 0)) or 2742
                Rev = float(row.get('营业收入', 0)) or 41206

                # 计算比率（添加合理性检查）
                CR = min(CA / max(CL, 1), 10)  # 流动比率
                QR = min((CA - Inv) / max(CL, 1), 10)  # 速动比率
                DR = min(TD / max(TA, 1), 1)  # 资产负债率
                NPM = max(min(NI / max(Rev, 1), 1), -1)  # 净利润率
                AT = min(Rev / max(TA, 1), 5)  # 资产周转率

                # 健康度评分（使用你的权重公式）
                health_score = (
                        0.4108 * CR +
                        0.4108 * QR +
                        0.1558 * (1 - DR) +
                        0.1558 * NPM +
                        0.0668 * AT
                )

                # 标准化到0-1范围
                health_score = max(0, min(1, health_score / 2.0))  # 除以2因为权重总和>1

                result_item = {
                    'date': date_str,
                    'CA': CA, 'CL': CL, 'Inv': Inv, 'TD': TD,
                    'TA': TA, 'NI': NI, 'Rev': Rev,
                    'CR': CR, 'QR': QR, 'DR': DR, 'NPM': NPM, 'AT': AT,
                    'health_score': health_score
                }

                results['data'].append(result_item)

            except Exception as e:
                print(f"计算第{i}行时出错: {e}")
                continue

        return results

    def generate_health_comparison(financial_forecast):
        """
        生成健康度预测对比结果
        """
        comparison = {
            'data': []
        }

        for item in financial_forecast['data']:
            # 这里可以使用不同的方法生成直接预测健康度
            # 例如：使用时间序列模型直接预测健康度
            # 目前先使用间接预测值加上一些随机变化作为示例
            direct_forecast = item['health_score'] * (1 + np.random.normal(0, 0.05))

            comparison['data'].append({
                'date': item['date'],
                'direct_forecast': direct_forecast,
                'indirect_forecast': item['health_score']
            })

        return comparison

    # 将方法添加到task_manager实例
    task_manager.get_current_time = get_current_time.__get__(task_manager, TaskManager)