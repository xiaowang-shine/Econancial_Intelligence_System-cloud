# report_generator.py
from datetime import datetime
from io import BytesIO

import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from typing import Dict, Any, List


def generate_financial_report(result: Dict[str, Any], task_id: str) -> BytesIO:
    """
    生成财务预测报告PDF - 完全修复版本（解决乱码问题）
    """
    try:
        from io import BytesIO
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.lib.fonts import addMapping
        import os

        # 注册中文字体 - 解决乱码问题
        try:
            # 尝试使用系统中文字体
            font_paths = [
                '/System/Library/Fonts/PingFang.ttc',  # macOS
                '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',  # Linux
                'C:/Windows/Fonts/simhei.ttf',  # Windows
                'C:/Windows/Fonts/simsun.ttc',  # Windows
            ]

            chinese_font = None
            for font_path in font_paths:
                if os.path.exists(font_path):
                    chinese_font = font_path
                    break

            if chinese_font:
                pdfmetrics.registerFont(TTFont('ChineseFont', chinese_font))
            else:
                # 使用默认字体，但设置编码
                from reportlab.pdfbase import pdfmetrics
                from reportlab.pdfbase.cidfonts import UnicodeCIDFont
                pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
        except:
            # 如果字体注册失败，使用默认处理
            pass

        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72,
                                topMargin=72, bottomMargin=18)

        styles = getSampleStyleSheet()

        # 创建中文字体样式
        try:
            chinese_style = ParagraphStyle(
                'ChineseStyle',
                parent=styles['Normal'],
                fontName='ChineseFont',
                fontSize=10,
                leading=14,
            )
            title_style = ParagraphStyle(
                'ChineseTitle',
                parent=styles['Title'],
                fontName='ChineseFont',
                fontSize=16,
                leading=24,
                alignment=1,  # 居中
            )
            heading_style = ParagraphStyle(
                'ChineseHeading',
                parent=styles['Heading2'],
                fontName='ChineseFont',
                fontSize=14,
                leading=20,
            )
        except:
            # 如果字体设置失败，使用默认样式
            chinese_style = styles['Normal']
            title_style = styles['Title']
            heading_style = styles['Heading2']

        story = []

        # 标题
        story.append(Paragraph("企业财务预测报告", title_style))
        story.append(Spacer(1, 20))

        # 报告信息
        current_time = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
        story.append(Paragraph(f"生成时间: {current_time}", chinese_style))
        story.append(Paragraph(f"任务ID: {task_id}", chinese_style))
        story.append(Spacer(1, 12))

        # 处理财务预测数据
        forecast_data = []
        data_source = ""

        if 'financial_forecast' in result and result['financial_forecast']:
            forecast_data = result['financial_forecast'].get('data', [])
            data_source = "核心算法预测"
        elif 'forecast' in result and result['forecast']:
            forecast_data = result['forecast']
            data_source = "训练模型预测"

        story.append(Paragraph(f"数据来源: {data_source}", heading_style))
        story.append(Spacer(1, 12))

        if forecast_data:
            # 预测概述
            story.append(Paragraph("未来12个月财务预测概述", heading_style))

            # 计算关键指标
            df_forecast = pd.DataFrame(forecast_data)
            avg_revenue = df_forecast['Rev'].mean() if 'Rev' in df_forecast.columns else 0
            avg_profit = df_forecast['NI'].mean() if 'NI' in df_forecast.columns else 0
            avg_health = df_forecast['health_score'].mean() if 'health_score' in df_forecast.columns else 0

            overview_text = f"""
            预测期间营业收入平均值: {avg_revenue:,.2f} 万元<br/>
            预测期间净利润平均值: {avg_profit:,.2f} 万元<br/>
            平均健康度评分: {avg_health:.4f}<br/>
            预测显示企业整体财务状况保持稳定趋势。
            """
            story.append(Paragraph(overview_text, chinese_style))
            story.append(Spacer(1, 12))

            # 预测表格（前6行，避免表格过大）
            story.append(Paragraph("财务预测数据（前6个月）", heading_style))

            # 准备表格数据
            table_data = [['日期', '营业收入(万元)', '净利润(万元)', '健康度评分']]

            for i, item in enumerate(forecast_data[:6]):
                date_str = str(item.get('date', ''))[:10]  # 取前10个字符，避免长日期
                revenue = item.get('Rev', item.get('value', 0))
                profit = item.get('NI', 0)
                health = item.get('health_score', 0)

                table_data.append([
                    date_str,
                    f"{revenue:,.2f}",
                    f"{profit:,.2f}",
                    f"{health:.4f}"
                ])

            # 创建表格
            table = Table(table_data, colWidths=[80, 100, 100, 80])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'ChineseFont'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8F9FA')),
                ('FONTNAME', (0, 1), (-1, -1), 'ChineseFont'),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#CCCCCC'))
            ]))

            story.append(table)
            story.append(Spacer(1, 12))

        # 生成专业建议
        story.append(Paragraph("专业发展建议", heading_style))
        suggestions = generate_detailed_suggestions(result)

        for i, suggestion in enumerate(suggestions, 1):
            suggestion_text = f"<b>建议{i}:</b> {suggestion}"
            story.append(Paragraph(suggestion_text, chinese_style))
            story.append(Spacer(1, 8))

        # 风险提示
        story.append(Paragraph("风险提示", heading_style))
        risk_text = """
        本预测基于历史数据和统计模型，实际结果可能受市场环境、政策变化、企业经营策略等多种因素影响。<br/>
        建议结合实际情况进行综合判断，本报告仅供参考。
        """
        story.append(Paragraph(risk_text, chinese_style))

        # 构建文档
        doc.build(story)
        buffer.seek(0)
        return buffer

    except Exception as e:
        print(f"生成PDF报告失败: {e}")
        return generate_error_report(f"生成报告时发生错误: {str(e)}")


def generate_error_report(error_msg: str) -> BytesIO:
    """生成错误报告PDF"""
    from io import BytesIO
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("报告生成错误", styles['Title']))
    story.append(Spacer(1, 20))
    story.append(Paragraph("抱歉，生成报告时出现错误:", styles['Normal']))
    story.append(Paragraph(error_msg, styles['Normal']))
    story.append(Spacer(1, 12))
    story.append(Paragraph("建议措施:", styles['Heading2']))
    story.append(Paragraph("1. 检查数据文件格式是否正确", styles['Normal']))
    story.append(Paragraph("2. 重新上传数据文件", styles['Normal']))
    story.append(Paragraph("3. 联系技术支持", styles['Normal']))

    doc.build(story)
    buffer.seek(0)
    return buffer


def generate_detailed_suggestions(result: Dict[str, Any]) -> List[str]:
    """
    根据预测结果生成详细的发展建议
    """
    suggestions = []

    try:
        # 提取财务数据
        financial_data = []
        if 'financial_forecast' in result and result['financial_forecast']:
            financial_data = result['financial_forecast'].get('data', [])

        if not financial_data:
            return ["无法生成建议：缺少财务预测数据"]

        df = pd.DataFrame(financial_data)

        # 计算关键指标趋势
        avg_health = df['health_score'].mean() if 'health_score' in df.columns else 0
        avg_revenue = df['Rev'].mean() if 'Rev' in df.columns else 0
        avg_profit = df['NI'].mean() if 'NI' in df.columns else 0
        avg_debt_ratio = df['DR'].mean() if 'DR' in df.columns else 0
        avg_current_ratio = df['CR'].mean() if 'CR' in df.columns else 0

        # 分析趋势
        revenue_trend = "稳定"
        if len(df) > 1:
            first_rev = df['Rev'].iloc[0] if 'Rev' in df.columns else 0
            last_rev = df['Rev'].iloc[-1] if 'Rev' in df.columns else 0
            revenue_growth = (last_rev - first_rev) / first_rev if first_rev > 0 else 0
            if revenue_growth > 0.1:
                revenue_trend = "快速增长"
            elif revenue_growth > 0.05:
                revenue_trend = "稳步增长"
            elif revenue_growth < -0.05:
                revenue_trend = "有所下滑"

        # 生成基于健康度评分的建议
        if avg_health >= 0.8:
            suggestions.append("企业财务状况优秀，健康度评分较高。建议继续保持当前经营策略，可适当扩大投资规模。")
        elif avg_health >= 0.6:
            suggestions.append("企业财务状况良好，但有提升空间。建议优化成本结构，提高资产使用效率。")
        else:
            suggestions.append("企业财务状况需要关注。建议加强现金流管理，控制负债规模。")

        # 基于营收趋势的建议
        if revenue_trend == "快速增长":
            suggestions.append("营业收入呈现快速增长趋势，市场需求旺盛。建议加大市场投入，扩大市场份额。")
        elif revenue_trend == "稳步增长":
            suggestions.append("营业收入稳步增长，经营策略有效。建议维持当前发展节奏，注重盈利质量。")
        elif revenue_trend == "有所下滑":
            suggestions.append("营业收入出现下滑，需关注市场变化。建议分析下滑原因，调整产品策略。")

        # 基于负债率的建议
        if avg_debt_ratio > 0.7:
            suggestions.append("资产负债率偏高，财务风险较大。建议优先偿还高成本债务，优化资本结构。")
        elif avg_debt_ratio > 0.5:
            suggestions.append("资产负债率处于合理区间上限。建议控制新增负债，加强应收账款管理。")
        else:
            suggestions.append("资产负债率合理，财务结构稳健。可考虑适度利用财务杠杆促进发展。")

        # 基于流动比率的建议
        if avg_current_ratio < 1.5:
            suggestions.append("流动比率偏低，短期偿债能力需关注。建议加强流动资金管理，保持适度现金储备。")
        elif avg_current_ratio > 3:
            suggestions.append("流动比率偏高，可能存在资金利用效率不高的问题。建议优化资金配置。")
        else:
            suggestions.append("流动比率处于健康区间，短期偿债能力良好。")

        # 基于盈利能力的建议
        avg_profit_margin = avg_profit / avg_revenue if avg_revenue > 0 else 0
        if avg_profit_margin > 0.15:
            suggestions.append("盈利能力强劲，利润率较高。建议加大研发投入，巩固竞争优势。")
        elif avg_profit_margin > 0.08:
            suggestions.append("盈利能力良好。建议通过精细化管理和成本控制进一步提升利润空间。")
        else:
            suggestions.append("盈利能力有待提升。建议分析成本结构，寻找增收节支的机会。")

        # 添加具体行动建议
        suggestions.extend([
            "定期进行财务健康度评估，及时调整经营策略。",
            "建立财务预警机制，对关键指标进行监控。",
            "加强预算管理，确保资金使用效率。",
            "优化存货管理，减少资金占用。",
            "关注行业发展趋势，适时调整业务结构。"
        ])

    except Exception as e:
        suggestions = [f"生成建议时发生错误: {str(e)}", "请检查数据完整性后重新分析。"]

    return suggestions[:8]  # 返回前8条最重要的建议

def generate_dashboard_chart_data(result: Dict[str, Any], historical_data: List[float] = None,
                                  historical_labels: List[str] = None) -> Dict[str, Any]:
    """
    生成仪表板图表数据

    Args:
        result: 包含预测结果的字典
        historical_data: 历史数据（可选）
        historical_labels: 历史数据标签（可选）

    Returns:
        图表数据字典
    """
    model_type = result['meta']['model']
    fluctuation = result['meta']['fluctuation']

    # 如果没有提供历史数据，使用默认值
    if historical_data is None:
        historical_data = [32983.99, 34962.43, 36645.53, 37723.82, 39898.37, 41206.21]

    if historical_labels is None:
        historical_labels = ["2024-07-31", "2024-08-31", "2024-09-30",
                             "2024-10-31", "2024-11-30", "2024-12-31"]

    # 预测数据
    pred_labels = [p['date'] for p in result['forecast']]
    pred_data = [p['value'] for p in result['forecast']]
    lower_data = [p['lower'] for p in result['forecast']]
    upper_data = [p['upper'] for p in result['forecast']]

    # 合并历史数据和预测数据
    full_labels = historical_labels + pred_labels
    hist_full = historical_data + [None] * len(pred_labels)
    pred_full = [None] * len(historical_labels) + pred_data
    lower_full = [None] * len(historical_labels) + lower_data
    upper_full = [None] * len(historical_labels) + upper_data

    # 主图表数据
    chart_data = {
        'type': 'line',
        'data': {
            'labels': full_labels,
            'datasets': [
                {'label': '历史营业收入', 'data': hist_full, 'borderColor': '#1E88E5', 'fill': False},
                {'label': '预测营业收入', 'data': pred_full, 'borderColor': '#EF5350', 'fill': False},
                {'label': '预测下限', 'data': lower_full, 'borderColor': '#EF5350', 'borderDash': [5, 5],
                 'fill': False},
                {'label': '预测上限', 'data': upper_full, 'borderColor': '#4CAF50', 'borderDash': [5, 5],
                 'fill': False},
            ]
        },
        'options': {
            'responsive': True,
            'scales': {
                'y': {'beginAtZero': False, 'title': {'display': True, 'text': '营业收入 (万元)'}},
                'x': {'title': {'display': True, 'text': '日期'}}
            },
            'plugins': {
                'title': {'display': True, 'text': f"2024-2025 年营业收入趋势 ({model_type}, 波动 ≈{fluctuation:.2f}%)"}
            }
        }
    }

    # KPI饼图
    debt_ratio = result['kpi'].get('debt_ratio', 0) or 0
    kpi_chart = {
        'type': 'pie',
        'data': {
            'labels': ['负债占比', '资产占比'],
            'datasets': [{'data': [debt_ratio, 100 - debt_ratio], 'backgroundColor': ['#EF5350', '#4CAF50']}]
        },
        'options': {'responsive': True, 'plugins': {'title': {'display': True, 'text': f"资产负债率 ({debt_ratio}%)"}}}
    }

    # 特征重要性图表
    imp_chart = None
    if result['explain'].get('feature_importance'):
        imp_names = [i['name'] for i in result['explain']['feature_importance'][:5]]
        imp_values = [i['importance'] for i in result['explain']['feature_importance'][:5]]
        imp_chart = {
            'type': 'bar',
            'data': {
                'labels': imp_names,
                'datasets': [{
                    'label': '重要性',
                    'data': imp_values,
                    'backgroundColor': '#1E88E5'
                }]
            },
            'options': {
                'responsive': True,
                'plugins': {
                    'title': {'display': True, 'text': 'Top 5 特征重要性'}
                }
            }
        }

    return {
        'main_chart': chart_data,
        'kpi_chart': kpi_chart,
        'importance_chart': imp_chart
    }


def generate_simple_report(result: Dict[str, Any], format_type: str = 'text') -> str:
    """
    生成简版报告（文本格式）

    Args:
        result: 包含预测结果的字典
        format_type: 报告格式（'text' 或 'html'）

    Returns:
        报告文本
    """
    model_type = result['meta']['model']
    fluctuation = result['meta']['fluctuation']

    if format_type == 'html':
        report = f"<h1>企业财务预测报告</h1>"
        report += f"<p><strong>模型类型:</strong> {model_type}</p>"
        report += f"<p><strong>预测波动:</strong> {fluctuation:.2f}%</p>"

        # KPI部分
        report += "<h2>关键绩效指标</h2><ul>"
        for k, v in result['kpi'].items():
            if v is not None:
                display_name = k.replace('_', ' ').title()
                report += f"<li>{display_name}: {v:.2f}</li>"
        report += "</ul>"

        # 预测部分
        report += "<h2>未来12个月预测</h2><table border='1'><tr><th>日期</th><th>预测值</th><th>下限</th><th>上限</th></tr>"
        for p in result['forecast'][:10]:  # 只显示前10行
            report += f"<tr><td>{p['date']}</td><td>{p['value']:.2f}</td><td>{p['lower']:.2f}</td><td>{p['upper']:.2f}</td></tr>"
        report += "</table>"

        # 建议部分
        report += "<h2>优化建议</h2><ul>"
        for s in result['suggestions']:
            report += f"<li>{s['priority']}优先: {s['text']} (影响: {s['impact']})</li>"
        report += "</ul>"

    else:  # 文本格式
        report = "企业财务预测报告\n"
        report += "=" * 50 + "\n"
        report += f"模型类型: {model_type}\n"
        report += f"预测波动: {fluctuation:.2f}%\n\n"

        # KPI部分
        report += "关键绩效指标:\n"
        report += "-" * 20 + "\n"
        for k, v in result['kpi'].items():
            if v is not None:
                display_name = k.replace('_', ' ').title()
                report += f"{display_name}: {v:.2f}\n"
        report += "\n"

        # 预测部分
        report += "未来12个月预测:\n"
        report += "-" * 20 + "\n"
        report += "日期        预测值    下限    上限\n"
        for p in result['forecast'][:10]:  # 只显示前10行
            report += f"{p['date']}  {p['value']:8.2f}  {p['lower']:6.2f}  {p['upper']:6.2f}\n"
        report += "\n"

        # 建议部分
        report += "优化建议:\n"
        report += "-" * 20 + "\n"
        for s in result['suggestions']:
            report += f"{s['priority']}优先: {s['text']} (影响: {s['impact']})\n"

    return report


def export_report_to_file(result: Dict[str, Any], task_id: str,
                          file_format: str = 'pdf') -> BytesIO:
    """
    导出报告到文件

    Args:
        result: 包含预测结果的字典
        task_id: 任务ID，用于文件名
        file_format: 文件格式（'pdf', 'txt', 'html'）

    Returns:
        文件的BytesIO对象
    """
    if file_format == 'pdf':
        return generate_financial_report(result, task_id)
    elif file_format == 'html':
        report_text = generate_simple_report(result, 'html')
        buffer = BytesIO()
        buffer.write(report_text.encode('utf-8'))
        buffer.seek(0)
        return buffer
    else:  # 默认为文本格式
        report_text = generate_simple_report(result, 'text')
        buffer = BytesIO()
        buffer.write(report_text.encode('utf-8'))
        buffer.seek(0)
        return buffer

def export_data_to_excel(result, task_id):
    """将预测结果导出为Excel文件 - 修复版本"""
    try:
        from io import BytesIO
        import pandas as pd

        # 创建Excel写入器
        output = BytesIO()

        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # 财务预测数据
            if 'forecast' in result and result['forecast']:
                try:
                    df_forecast = pd.DataFrame(result['forecast'])
                    # 确保数值列是数字类型
                    numeric_cols = ['value', 'lower', 'upper']
                    for col in numeric_cols:
                        if col in df_forecast.columns:
                            df_forecast[col] = pd.to_numeric(df_forecast[col], errors='coerce').fillna(0)
                    df_forecast.to_excel(writer, sheet_name='财务预测', index=False)
                except Exception as e:
                    print(f"导出财务预测数据失败: {e}")

            # 健康度预测数据
            if 'health_forecast' in result and result['health_forecast']:
                try:
                    df_health = pd.DataFrame(result['health_forecast'])
                    df_health.to_excel(writer, sheet_name='健康度预测', index=False)
                except Exception as e:
                    print(f"导出健康度预测数据失败: {e}")

            # 如果是核心算法结果，处理不同的数据结构
            if 'financial_forecast' in result and result['financial_forecast']:
                try:
                    financial_data = result['financial_forecast'].get('data', [])
                    if financial_data:
                        df_financial = pd.DataFrame(financial_data)
                        df_financial.to_excel(writer, sheet_name='财务指标', index=False)
                except Exception as e:
                    print(f"导出财务指标数据失败: {e}")

            # 优化建议
            if 'suggestions' in result and result['suggestions']:
                try:
                    df_suggestions = pd.DataFrame(result['suggestions'])
                    df_suggestions.to_excel(writer, sheet_name='优化建议', index=False)
                except Exception as e:
                    print(f"导出优化建议失败: {e}")

            # KPI指标
            if 'kpi' in result:
                try:
                    kpi_data = {k: [v] for k, v in result['kpi'].items() if v is not None}
                    if kpi_data:
                        df_kpi = pd.DataFrame(kpi_data)
                        df_kpi.to_excel(writer, sheet_name='关键指标', index=False)
                except Exception as e:
                    print(f"导出KPI指标失败: {e}")

        output.seek(0)
        return output

    except Exception as e:
        print(f"导出Excel失败: {e}")
        # 返回一个空的Excel文件作为兜底
        return create_fallback_excel()


def export_data_to_csv(result, task_id):
    """将预测结果导出为CSV文件 - 修复编码问题版本"""
    try:
        import pandas as pd
        from io import StringIO
        import csv

        output = StringIO()
        writer = csv.writer(output)

        # 使用英文列名避免编码问题
        if 'financial_forecast' in result and result['financial_forecast']:
            financial_data = result['financial_forecast'].get('data', [])
            if financial_data:
                writer.writerow(['=== Financial Forecast Results ==='])
                df_financial = pd.DataFrame(financial_data)

                # 使用英文列名
                column_mapping = {
                    'date': 'Date', 'CA': 'Current Assets', 'CL': 'Current Liabilities',
                    'Inv': 'Inventory', 'TD': 'Total Debt', 'TA': 'Total Assets',
                    'NI': 'Net Profit', 'Rev': 'Revenue', 'CR': 'Current Ratio',
                    'QR': 'Quick Ratio', 'DR': 'Debt Ratio', 'NPM': 'Net Profit Margin',
                    'AT': 'Asset Turnover', 'health_score': 'Health Score'
                }
                df_financial = df_financial.rename(columns=column_mapping)

                # 只保留需要的列
                keep_columns = ['Date', 'Revenue', 'Net Profit', 'Current Ratio',
                                'Quick Ratio', 'Debt Ratio', 'Health Score']
                available_columns = [col for col in keep_columns if col in df_financial.columns]
                df_financial = df_financial[available_columns]

                # 格式化数值
                for col in df_financial.columns:
                    if col in ['Revenue', 'Net Profit']:
                        df_financial[col] = df_financial[col].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else "0.00")
                    elif col in ['Current Ratio', 'Quick Ratio', 'Debt Ratio', 'Health Score']:
                        df_financial[col] = df_financial[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "0.0000")

                # 写入CSV
                df_financial.to_csv(output, mode='a', index=False, encoding='utf-8')
                writer.writerow([])

        if 'health_comparison' in result and result['health_comparison']:
            health_data = result['health_comparison'].get('data', [])
            if health_data:
                writer.writerow(['=== Health Score Comparison ==='])
                df_health = pd.DataFrame(health_data)

                # 使用英文列名
                health_mapping = {
                    'date': 'Date',
                    'direct_forecast': 'Direct Forecast',
                    'indirect_forecast': 'Indirect Forecast'
                }
                df_health = df_health.rename(columns=health_mapping)

                # 格式化健康度评分
                for col in ['Direct Forecast', 'Indirect Forecast']:
                    if col in df_health.columns:
                        df_health[col] = df_health[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "0.0000")

                df_health.to_csv(output, mode='a', index=False, encoding='utf-8')
                writer.writerow([])

        # 添加统计摘要（英文）
        if 'financial_forecast' in result and result['financial_forecast']:
            financial_data = result['financial_forecast'].get('data', [])
            if financial_data:
                df = pd.DataFrame(financial_data)
                writer.writerow(['=== Summary Statistics ==='])
                summary_data = [
                    ['Metric', 'Average', 'Max', 'Min'],
                    ['Revenue', f"{df['Rev'].mean():,.2f}", f"{df['Rev'].max():,.2f}", f"{df['Rev'].min():,.2f}"],
                    ['Net Profit', f"{df['NI'].mean():,.2f}", f"{df['NI'].max():,.2f}", f"{df['NI'].min():,.2f}"],
                    ['Health Score', f"{df['health_score'].mean():.4f}", f"{df['health_score'].max():.4f}",
                     f"{df['health_score'].min():.4f}"]
                ]
                for row in summary_data:
                    writer.writerow(row)

        # 获取CSV内容并编码为UTF-8字节
        csv_content = output.getvalue()
        return csv_content.encode('utf-8')

    except Exception as e:
        print(f"CSV export failed: {e}")
        # 返回简单的错误信息CSV（英文）
        error_content = "Error,Export process failed\nSuggestion,Please try again or contact support"
        return error_content.encode('utf-8')


def create_fallback_excel():
    """创建兜底Excel文件"""
    from io import BytesIO
    import pandas as pd

    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # 创建一个简单的错误提示工作表
        error_df = pd.DataFrame({
            '错误信息': ['导出过程中发生错误', '请检查数据格式或联系技术支持'],
            '建议': ['重新上传数据文件', '检查文件格式是否符合要求']
        })
        error_df.to_excel(writer, sheet_name='错误提示', index=False)

    output.seek(0)
    return output