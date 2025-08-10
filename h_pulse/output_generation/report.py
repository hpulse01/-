"""
报告生成模块
生成PDF报告、时间轴可视化、图表等
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import structlog

# PDF生成
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, KeepTogether, Frame, PageTemplate
)
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.widgets.markers import makeMarker
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# 可视化
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 其他
from ..utils.settings import get_settings, BrandColors
from ..utils.crypto_anchor import sign_prediction, anchor_to_blockchain

logger = structlog.get_logger()


class ReportGenerator:
    """报告生成器"""
    
    def __init__(self):
        self.settings = get_settings()
        self.colors = BrandColors()
        
        # 注册中文字体（如果有的话）
        try:
            # 尝试注册思源黑体或其他中文字体
            font_path = self.settings.assets_dir / "fonts" / "NotoSansSC-Regular.ttf"
            if font_path.exists():
                pdfmetrics.registerFont(TTFont('NotoSansSC', str(font_path)))
                self.chinese_font = 'NotoSansSC'
            else:
                self.chinese_font = 'Helvetica'  # 降级到默认字体
        except:
            self.chinese_font = 'Helvetica'
            logger.warning("未能加载中文字体，使用默认字体")
        
        # 样式
        self.styles = self._create_styles()
    
    def _create_styles(self) -> Dict[str, ParagraphStyle]:
        """创建报告样式"""
        styles = getSampleStyleSheet()
        
        # 标题样式
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontName=self.chinese_font,
            fontSize=24,
            textColor=colors.HexColor(self.colors.QUANTUM_BLUE),
            spaceAfter=30,
            alignment=1  # 居中
        )
        
        # 副标题样式
        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Heading2'],
            fontName=self.chinese_font,
            fontSize=16,
            textColor=colors.HexColor(self.colors.MYSTIC_PURPLE),
            spaceAfter=12
        )
        
        # 正文样式
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['BodyText'],
            fontName=self.chinese_font,
            fontSize=10,
            leading=14,
            textColor=colors.HexColor('#333333')
        )
        
        # 高亮样式
        highlight_style = ParagraphStyle(
            'Highlight',
            parent=body_style,
            backColor=colors.HexColor('#FFF9C4'),
            borderColor=colors.HexColor(self.colors.PREDICTION_GREEN),
            borderWidth=1,
            borderPadding=3
        )
        
        return {
            'title': title_style,
            'subtitle': subtitle_style,
            'body': body_style,
            'highlight': highlight_style
        }
    
    def generate_report(self, prediction_data: Dict[str, Any], 
                       output_path: Path) -> Path:
        """
        生成完整的PDF报告
        
        Args:
            prediction_data: 预测结果数据
            output_path: 输出路径
            
        Returns:
            生成的PDF文件路径
        """
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 创建PDF文档
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            rightMargin=2*cm,
            leftMargin=2*cm,
            topMargin=2*cm,
            bottomMargin=2*cm
        )
        
        # 构建内容
        story = []
        
        # 封面
        story.extend(self._create_cover_page(prediction_data))
        story.append(PageBreak())
        
        # 目录
        story.extend(self._create_table_of_contents())
        story.append(PageBreak())
        
        # 执行摘要
        story.extend(self._create_executive_summary(prediction_data))
        story.append(PageBreak())
        
        # 生命轨迹时间轴
        story.extend(self._create_timeline_section(prediction_data))
        story.append(PageBreak())
        
        # 详细事件分析
        story.extend(self._create_events_analysis(prediction_data))
        story.append(PageBreak())
        
        # 特征贡献分析
        story.extend(self._create_feature_analysis(prediction_data))
        story.append(PageBreak())
        
        # 建议与对策
        story.extend(self._create_recommendations(prediction_data))
        story.append(PageBreak())
        
        # 技术附录
        story.extend(self._create_technical_appendix(prediction_data))
        story.append(PageBreak())
        
        # 签名与验证
        story.extend(self._create_signature_section(prediction_data))
        
        # 生成PDF
        doc.build(story, onFirstPage=self._add_header_footer,
                 onLaterPages=self._add_header_footer)
        
        logger.info("PDF报告已生成", path=str(output_path))
        
        return output_path
    
    def _create_cover_page(self, data: Dict[str, Any]) -> List:
        """创建封面"""
        elements = []
        
        # 标题
        elements.append(Spacer(1, 3*inch))
        elements.append(Paragraph(
            "H-Pulse 量子生命轨迹预测报告",
            self.styles['title']
        ))
        
        elements.append(Spacer(1, 0.5*inch))
        
        # 副标题
        elements.append(Paragraph(
            "Precision · Uniqueness · Irreversibility",
            self.styles['subtitle']
        ))
        
        elements.append(Spacer(1, 1*inch))
        
        # 用户信息
        birth_info = data.get('birth_info', {})
        info_text = f"""
        <para align="center">
        用户ID: {data.get('user_id', 'Anonymous')}<br/>
        出生时间: {birth_info.get('birth_datetime', 'Unknown')}<br/>
        预测时间: {data.get('prediction_time', datetime.now().isoformat())}<br/>
        </para>
        """
        elements.append(Paragraph(info_text, self.styles['body']))
        
        # Logo占位
        elements.append(Spacer(1, 2*inch))
        
        # 版权信息
        elements.append(Paragraph(
            f"© {datetime.now().year} H-Pulse Quantum Prediction System. All rights reserved.",
            ParagraphStyle('Footer', parent=self.styles['body'], 
                         fontSize=8, alignment=1, textColor=colors.grey)
        ))
        
        return elements
    
    def _create_table_of_contents(self) -> List:
        """创建目录"""
        elements = []
        
        elements.append(Paragraph("目录", self.styles['title']))
        elements.append(Spacer(1, 0.5*inch))
        
        toc_data = [
            ["章节", "页码"],
            ["1. 执行摘要", "3"],
            ["2. 生命轨迹时间轴", "4"],
            ["3. 详细事件分析", "6"],
            ["4. 特征贡献分析", "10"],
            ["5. 建议与对策", "12"],
            ["6. 技术附录", "14"],
            ["7. 签名与验证", "16"]
        ]
        
        toc_table = Table(toc_data, colWidths=[4*inch, 1*inch])
        toc_table.setStyle(TableStyle([
            ('FONT', (0, 0), (-1, 0), self.chinese_font, 12),
            ('FONT', (0, 1), (-1, -1), self.chinese_font, 10),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor(self.colors.QUANTUM_BLUE)),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ('LINEBELOW', (0, 0), (-1, 0), 1, colors.HexColor(self.colors.QUANTUM_BLUE)),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F5F5F5')])
        ]))
        
        elements.append(toc_table)
        
        return elements
    
    def _create_executive_summary(self, data: Dict[str, Any]) -> List:
        """创建执行摘要"""
        elements = []
        
        elements.append(Paragraph("执行摘要", self.styles['title']))
        elements.append(Spacer(1, 0.3*inch))
        
        # 关键发现
        trajectory = data.get('life_trajectory', {})
        summary = trajectory.get('timeline_summary', '暂无摘要')
        trend = trajectory.get('overall_trend', {})
        
        summary_text = f"""
        <para>
        基于量子叠加态模拟和深度学习分析，我们对您的生命轨迹进行了全面预测。
        以下是关键发现：<br/><br/>
        
        <b>整体趋势：</b>{trend.get('description', '运势平稳')}<br/><br/>
        
        <b>重要事件预览：</b><br/>
        {summary}<br/><br/>
        
        <b>置信度评估：</b><br/>
        整体预测置信度：{data.get('confidence_metrics', {}).get('overall_confidence', 0):.1%}<br/>
        模型不确定性：{data.get('confidence_metrics', {}).get('prediction_uncertainty', 0):.3f}<br/>
        </para>
        """
        
        elements.append(Paragraph(summary_text, self.styles['body']))
        
        # 关键时期
        elements.append(Spacer(1, 0.2*inch))
        elements.append(Paragraph("关键时期", self.styles['subtitle']))
        
        key_periods = trajectory.get('key_periods', [])
        if key_periods:
            period_data = [["时期类型", "描述", "起始时间", "重要性"]]
            for period in key_periods:
                period_data.append([
                    period.get('type', ''),
                    period.get('description', ''),
                    period.get('start_date', '')[:10],
                    period.get('importance', '')
                ])
            
            period_table = Table(period_data, colWidths=[1.5*inch, 2*inch, 1.5*inch, 1*inch])
            period_table.setStyle(TableStyle([
                ('FONT', (0, 0), (-1, -1), self.chinese_font, 9),
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(self.colors.QUANTUM_BLUE)),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
            ]))
            
            elements.append(period_table)
        
        return elements
    
    def _create_timeline_section(self, data: Dict[str, Any]) -> List:
        """创建时间轴部分"""
        elements = []
        
        elements.append(Paragraph("生命轨迹时间轴", self.styles['title']))
        elements.append(Spacer(1, 0.3*inch))
        
        # 生成时间轴图表
        timeline_path = self._generate_timeline_chart(data)
        if timeline_path and timeline_path.exists():
            elements.append(Image(str(timeline_path), width=6*inch, height=4*inch))
        
        elements.append(Spacer(1, 0.2*inch))
        
        # 事件列表
        events = data.get('life_trajectory', {}).get('events', [])
        
        elements.append(Paragraph("预测事件详情", self.styles['subtitle']))
        
        for i, event in enumerate(events[:10]):  # 最多显示10个事件
            event_text = f"""
            <para>
            <b>事件{i+1}：{event.get('description', '未知事件')}</b><br/>
            类型：{event.get('type', 'unknown')}<br/>
            预期时间：{event.get('expected_date', '')[:10]}<br/>
            概率：{event.get('probability', 0):.1%}<br/>
            影响程度：{event.get('impact', 'medium')}<br/>
            </para>
            """
            
            elements.append(Paragraph(event_text, self.styles['body']))
            elements.append(Spacer(1, 0.1*inch))
        
        return elements
    
    def _create_events_analysis(self, data: Dict[str, Any]) -> List:
        """创建详细事件分析"""
        elements = []
        
        elements.append(Paragraph("详细事件分析", self.styles['title']))
        elements.append(Spacer(1, 0.3*inch))
        
        events = data.get('life_trajectory', {}).get('events', [])
        
        # 按类型分组
        events_by_type = {}
        for event in events:
            event_type = event.get('type', 'unknown')
            if event_type not in events_by_type:
                events_by_type[event_type] = []
            events_by_type[event_type].append(event)
        
        # 为每种类型创建分析
        type_names = {
            'career': '事业发展',
            'relationship': '感情关系',
            'health': '健康状况',
            'wealth': '财富运势',
            'education': '学习成长',
            'travel': '出行变动'
        }
        
        for event_type, type_events in events_by_type.items():
            type_name = type_names.get(event_type, event_type)
            
            elements.append(Paragraph(f"{type_name}分析", self.styles['subtitle']))
            
            # 统计信息
            avg_prob = np.mean([e.get('probability', 0) for e in type_events])
            
            analysis_text = f"""
            <para>
            在{type_name}方面，共预测到{len(type_events)}个相关事件，
            平均发生概率为{avg_prob:.1%}。<br/><br/>
            
            主要事件包括：<br/>
            </para>
            """
            
            elements.append(Paragraph(analysis_text, self.styles['body']))
            
            # 事件详情表
            event_data = [["事件描述", "时间", "概率", "建议"]]
            for event in type_events[:3]:  # 每类最多3个
                suggestions = event.get('suggestions', [])
                suggestion_text = suggestions[0] if suggestions else "保持关注"
                
                event_data.append([
                    event.get('description', '')[:30],
                    event.get('expected_date', '')[:10],
                    f"{event.get('probability', 0):.1%}",
                    suggestion_text
                ])
            
            event_table = Table(event_data, colWidths=[2*inch, 1.2*inch, 0.8*inch, 2*inch])
            event_table.setStyle(TableStyle([
                ('FONT', (0, 0), (-1, -1), self.chinese_font, 9),
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(self.colors.NEURAL_RED)),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('VALIGN', (0, 0), (-1, -1), 'TOP')
            ]))
            
            elements.append(event_table)
            elements.append(Spacer(1, 0.2*inch))
        
        return elements
    
    def _create_feature_analysis(self, data: Dict[str, Any]) -> List:
        """创建特征贡献分析"""
        elements = []
        
        elements.append(Paragraph("预测特征贡献分析", self.styles['title']))
        elements.append(Spacer(1, 0.3*inch))
        
        # 特征重要性
        feature_importance = data.get('feature_importance', {})
        
        explanation_text = """
        <para>
        本预测综合了多种数据源和分析方法，以下是各特征对预测结果的贡献度分析：
        </para>
        """
        
        elements.append(Paragraph(explanation_text, self.styles['body']))
        elements.append(Spacer(1, 0.2*inch))
        
        # 生成特征重要性图表
        importance_chart_path = self._generate_feature_importance_chart(feature_importance)
        if importance_chart_path and importance_chart_path.exists():
            elements.append(Image(str(importance_chart_path), width=5*inch, height=3*inch))
        
        # 特征说明
        feature_descriptions = {
            'astro': '天文因素：包括出生时的天体位置、季节、太阳月亮周期等',
            'eastern': '东方命理：四柱八字、紫微斗数等传统预测体系',
            'western': '西方占星：行星相位、宫位系统、过境影响等',
            'quantum': '量子特征：基于量子叠加态和纠缠的概率分布'
        }
        
        elements.append(Spacer(1, 0.2*inch))
        
        for feat, desc in feature_descriptions.items():
            importance = feature_importance.get(feat, 0)
            feat_text = f"""
            <para>
            <b>{desc}</b><br/>
            贡献度：{importance:.1%}<br/>
            </para>
            """
            elements.append(Paragraph(feat_text, self.styles['body']))
            elements.append(Spacer(1, 0.1*inch))
        
        return elements
    
    def _create_recommendations(self, data: Dict[str, Any]) -> List:
        """创建建议与对策"""
        elements = []
        
        elements.append(Paragraph("个性化建议与对策", self.styles['title']))
        elements.append(Spacer(1, 0.3*inch))
        
        # 总体建议
        trend = data.get('life_trajectory', {}).get('overall_trend', {})
        
        general_text = f"""
        <para>
        基于您的整体运势趋势（{trend.get('description', '平稳发展')}），
        我们为您提供以下建议：
        </para>
        """
        
        elements.append(Paragraph(general_text, self.styles['body']))
        elements.append(Spacer(1, 0.2*inch))
        
        # 分类建议
        recommendations = {
            'career': [
                "把握机遇期，主动寻求发展",
                "提升专业技能，增强竞争力",
                "建立良好的职场人际关系"
            ],
            'relationship': [
                "保持开放心态，真诚待人",
                "加强沟通，理解对方需求",
                "平衡个人空间与亲密关系"
            ],
            'health': [
                "建立规律的作息习惯",
                "适度运动，保持身心平衡",
                "定期体检，预防胜于治疗"
            ],
            'wealth': [
                "合理规划，避免冲动消费",
                "分散投资，控制风险",
                "持续学习理财知识"
            ]
        }
        
        # 根据事件类型给出针对性建议
        events = data.get('life_trajectory', {}).get('events', [])
        important_types = set()
        for event in events:
            if event.get('probability', 0) > 0.3:
                important_types.add(event.get('type'))
        
        for event_type in important_types:
            if event_type in recommendations:
                type_name = {
                    'career': '事业发展',
                    'relationship': '感情关系',
                    'health': '健康管理',
                    'wealth': '财富管理'
                }.get(event_type, event_type)
                
                elements.append(Paragraph(f"{type_name}建议", self.styles['subtitle']))
                
                for rec in recommendations[event_type]:
                    elements.append(Paragraph(f"• {rec}", self.styles['body']))
                
                elements.append(Spacer(1, 0.15*inch))
        
        # 风险提示
        elements.append(Paragraph("风险提示", self.styles['subtitle']))
        
        risk_text = """
        <para>
        • 本预测基于当前可获得的信息和模型分析，未来可能因各种因素发生变化<br/>
        • 建议将预测结果作为参考，结合实际情况做出决策<br/>
        • 保持积极心态，主动创造有利条件<br/>
        • 定期回顾和调整计划，适应环境变化<br/>
        </para>
        """
        
        elements.append(Paragraph(risk_text, self.styles['body']))
        
        return elements
    
    def _create_technical_appendix(self, data: Dict[str, Any]) -> List:
        """创建技术附录"""
        elements = []
        
        if not self.settings.report_include_technical_details:
            return elements
        
        elements.append(Paragraph("技术附录", self.styles['title']))
        elements.append(Spacer(1, 0.3*inch))
        
        # 模型信息
        elements.append(Paragraph("预测模型信息", self.styles['subtitle']))
        
        model_info = f"""
        <para>
        模型版本：{self.settings.model_name}<br/>
        预测时间：{data.get('prediction_time', 'Unknown')}<br/>
        置信度：{data.get('confidence_metrics', {}).get('overall_confidence', 0):.2f}<br/>
        不确定性：{data.get('confidence_metrics', {}).get('prediction_uncertainty', 0):.3f}<br/>
        </para>
        """
        
        elements.append(Paragraph(model_info, self.styles['body']))
        elements.append(Spacer(1, 0.2*inch))
        
        # 数据源
        elements.append(Paragraph("数据源说明", self.styles['subtitle']))
        
        data_sources = """
        <para>
        1. <b>天文数据</b>：JPL DE440星历表，考虑岁差、章动、光行差修正<br/>
        2. <b>时间系统</b>：UTC/TT/TDB转换，ΔT修正，真太阳时计算<br/>
        3. <b>东方体系</b>：四柱八字（考虑真太阳时）、紫微斗数（十四主星+四化）<br/>
        4. <b>西方体系</b>：行星位置（日月水金火木土天海冥）、宫位系统（Placidus）<br/>
        5. <b>量子模拟</b>：基于Qiskit的量子叠加态模拟，蒙特卡洛后验细化<br/>
        6. <b>AI模型</b>：Transformer + Graph Neural Network多模态融合架构<br/>
        </para>
        """
        
        elements.append(Paragraph(data_sources, self.styles['body']))
        
        # 假设说明
        elements.append(Spacer(1, 0.2*inch))
        elements.append(Paragraph("关键假设", self.styles['subtitle']))
        
        assumptions = """
        <para>
        • 出生时间精确到分钟级别<br/>
        • 地理坐标基于WGS84大地测量系统<br/>
        • 量子态测量遵循玻恩规则<br/>
        • 历史数据具有预测未来的参考价值<br/>
        • 多模态特征之间存在有意义的相关性<br/>
        </para>
        """
        
        elements.append(Paragraph(assumptions, self.styles['body']))
        
        return elements
    
    def _create_signature_section(self, data: Dict[str, Any]) -> List:
        """创建签名与验证部分"""
        elements = []
        
        elements.append(Paragraph("数字签名与区块链验证", self.styles['title']))
        elements.append(Spacer(1, 0.3*inch))
        
        # 生成签名
        signature, public_key = sign_prediction(data)
        
        # 尝试区块链锚定
        anchor_result = anchor_to_blockchain(data)
        
        sig_text = f"""
        <para>
        <b>数字签名信息</b><br/>
        签名算法：Ed25519<br/>
        公钥：{public_key[:32]}...{public_key[-16:]}<br/>
        签名：{signature[:32]}...{signature[-16:]}<br/>
        </para>
        """
        
        elements.append(Paragraph(sig_text, self.styles['body']))
        elements.append(Spacer(1, 0.2*inch))
        
        if anchor_result:
            anchor_text = f"""
            <para>
            <b>区块链锚定信息</b><br/>
            网络：{anchor_result.get('network', 'Unknown')}<br/>
            交易哈希：{anchor_result.get('tx_hash', 'N/A')}<br/>
            区块高度：{anchor_result.get('block_number', 'N/A')}<br/>
            时间戳：{anchor_result.get('timestamp', 'N/A')}<br/>
            </para>
            """
            elements.append(Paragraph(anchor_text, self.styles['body']))
        else:
            elements.append(Paragraph(
                "区块链锚定：未启用或不可用",
                self.styles['body']
            ))
        
        # 验证说明
        elements.append(Spacer(1, 0.2*inch))
        elements.append(Paragraph("验证说明", self.styles['subtitle']))
        
        verify_text = """
        <para>
        本报告已通过数字签名技术确保内容的真实性和完整性。
        您可以使用提供的公钥和签名信息验证报告未被篡改。
        如已启用区块链锚定，可通过交易哈希在相应区块链浏览器中查询验证。
        </para>
        """
        
        elements.append(Paragraph(verify_text, self.styles['body']))
        
        return elements
    
    def _generate_timeline_chart(self, data: Dict[str, Any]) -> Optional[Path]:
        """生成时间轴图表"""
        try:
            events = data.get('life_trajectory', {}).get('events', [])
            if not events:
                return None
            
            # 准备数据
            dates = []
            probabilities = []
            descriptions = []
            types = []
            
            for event in events[:10]:  # 最多10个事件
                date_str = event.get('expected_date', '')
                if date_str:
                    dates.append(datetime.fromisoformat(date_str.replace('T', ' ')))
                    probabilities.append(event.get('probability', 0))
                    descriptions.append(event.get('description', '')[:20])
                    types.append(event.get('type', 'unknown'))
            
            # 创建时间轴图
            fig = go.Figure()
            
            # 为不同类型使用不同颜色
            type_colors = {
                'career': self.colors.QUANTUM_BLUE,
                'relationship': self.colors.NEURAL_RED,
                'health': self.colors.PREDICTION_GREEN,
                'wealth': self.colors.MYSTIC_PURPLE
            }
            
            for i, (date, prob, desc, evt_type) in enumerate(zip(dates, probabilities, descriptions, types)):
                color = type_colors.get(evt_type, self.colors.MEDIUM_GRAY)
                
                fig.add_trace(go.Scatter(
                    x=[date],
                    y=[i],
                    mode='markers+text',
                    marker=dict(size=prob*50, color=color, opacity=0.7),
                    text=f"{desc}<br>概率: {prob:.1%}",
                    textposition="top center",
                    name=evt_type,
                    showlegend=False
                ))
            
            # 添加时间轴线
            fig.add_shape(
                type="line",
                x0=min(dates),
                y0=-1,
                x1=max(dates),
                y1=-1,
                line=dict(color=self.colors.DEEP_SPACE_BLACK, width=2)
            )
            
            fig.update_layout(
                title="生命轨迹时间轴",
                xaxis_title="时间",
                yaxis_title="事件",
                height=400,
                template="plotly_white",
                font=dict(size=10)
            )
            
            # 保存图表
            output_path = self.settings.get_output_path("timeline_chart.png", "charts")
            fig.write_image(str(output_path))
            
            return output_path
            
        except Exception as e:
            logger.error("生成时间轴图表失败", error=str(e))
            return None
    
    def _generate_feature_importance_chart(self, importance: Dict[str, float]) -> Optional[Path]:
        """生成特征重要性图表"""
        try:
            if not importance:
                return None
            
            # 准备数据
            features = list(importance.keys())
            values = list(importance.values())
            
            # 创建饼图
            fig = go.Figure(data=[go.Pie(
                labels=features,
                values=values,
                hole=0.3,
                marker_colors=[
                    self.colors.QUANTUM_BLUE,
                    self.colors.NEURAL_RED,
                    self.colors.PREDICTION_GREEN,
                    self.colors.MYSTIC_PURPLE
                ]
            )])
            
            fig.update_layout(
                title="预测特征贡献度",
                height=300,
                template="plotly_white",
                font=dict(size=10)
            )
            
            # 保存图表
            output_path = self.settings.get_output_path("feature_importance.png", "charts")
            fig.write_image(str(output_path))
            
            return output_path
            
        except Exception as e:
            logger.error("生成特征重要性图表失败", error=str(e))
            return None
    
    def _add_header_footer(self, canvas, doc):
        """添加页眉页脚"""
        canvas.saveState()
        
        # 页眉
        canvas.setFont(self.chinese_font, 8)
        canvas.setFillColor(colors.HexColor(self.colors.MEDIUM_GRAY))
        canvas.drawString(inch, A4[1] - 0.5*inch, "H-Pulse Quantum Prediction System")
        canvas.drawRightString(A4[0] - inch, A4[1] - 0.5*inch, 
                              datetime.now().strftime("%Y-%m-%d"))
        
        # 页脚
        canvas.drawString(inch, 0.5*inch, "Precision · Uniqueness · Irreversibility")
        canvas.drawRightString(A4[0] - inch, 0.5*inch, f"Page {doc.page}")
        
        # 页脚线
        canvas.setStrokeColor(colors.HexColor(self.colors.QUANTUM_BLUE))
        canvas.setLineWidth(0.5)
        canvas.line(inch, 0.75*inch, A4[0] - inch, 0.75*inch)
        
        canvas.restoreState()


def generate_pdf_report(prediction_data: Dict[str, Any], 
                       output_filename: str = "prediction_report.pdf") -> Path:
    """生成PDF报告的便捷函数"""
    generator = ReportGenerator()
    output_path = generator.settings.get_output_path(output_filename, "reports")
    return generator.generate_report(prediction_data, output_path)


def create_timeline_visualization(events: List[Dict[str, Any]], 
                                output_path: Optional[Path] = None) -> Path:
    """创建时间轴可视化的便捷函数"""
    # 创建交互式时间轴
    fig = go.Figure()
    
    for event in events:
        date = datetime.fromisoformat(event['expected_date'])
        
        fig.add_trace(go.Scatter(
            x=[date],
            y=[event['probability']],
            mode='markers+text',
            marker=dict(
                size=20,
                color=event['probability'],
                colorscale='Viridis',
                showscale=True
            ),
            text=event['description'],
            textposition="top center"
        ))
    
    fig.update_layout(
        title="Life Trajectory Timeline",
        xaxis_title="Time",
        yaxis_title="Probability",
        hovermode='closest'
    )
    
    if output_path is None:
        output_path = Path("timeline.html")
    
    fig.write_html(str(output_path))
    return output_path


def create_feature_importance_chart(importance: Dict[str, float],
                                  output_path: Optional[Path] = None) -> Path:
    """创建特征重要性图表的便捷函数"""
    fig = px.bar(
        x=list(importance.values()),
        y=list(importance.keys()),
        orientation='h',
        title='Feature Importance',
        labels={'x': 'Importance', 'y': 'Feature'}
    )
    
    if output_path is None:
        output_path = Path("feature_importance.html")
    
    fig.write_html(str(output_path))
    return output_path


if __name__ == "__main__":
    # 测试报告生成
    
    # 模拟预测数据
    test_data = {
        'user_id': 'test_001',
        'birth_info': {
            'birth_datetime': '1990-01-01T12:00:00+08:00',
            'longitude': 116.4074,
            'latitude': 39.9042
        },
        'prediction_time': datetime.now().isoformat(),
        'life_trajectory': {
            'events': [
                {
                    'type': 'career',
                    'description': '职业发展机遇',
                    'expected_date': '2025-06-15T10:00:00',
                    'probability': 0.85,
                    'confidence': 0.8,
                    'impact': 'major',
                    'suggestions': ['把握机会', '提升技能']
                },
                {
                    'type': 'relationship',
                    'description': '重要感情发展',
                    'expected_date': '2025-09-20T18:00:00',
                    'probability': 0.72,
                    'confidence': 0.75,
                    'impact': 'major',
                    'suggestions': ['保持开放', '真诚沟通']
                }
            ],
            'timeline_summary': '2025年将迎来事业和感情的双重机遇',
            'overall_trend': {
                'trend': 'ascending',
                'description': '整体运势上升'
            },
            'key_periods': [
                {
                    'type': 'career_peak',
                    'description': '事业高峰期',
                    'start_date': '2025-06-01',
                    'end_date': '2025-12-31',
                    'importance': 'high'
                }
            ]
        },
        'feature_importance': {
            'astro': 0.3,
            'eastern': 0.25,
            'western': 0.25,
            'quantum': 0.2
        },
        'confidence_metrics': {
            'overall_confidence': 0.78,
            'prediction_uncertainty': 0.15,
            'model_version': 'h-pulse-v1.1'
        }
    }
    
    # 生成报告
    report_path = generate_pdf_report(test_data, "test_report.pdf")
    print(f"报告已生成: {report_path}")