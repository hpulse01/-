#!/usr/bin/env python3
"""
H-Pulse Quantum Prediction System
主入口程序和CLI

使用方法:
    python main.py demo              # 运行演示
    python main.py api               # 启动API服务器
    python main.py train             # 训练AI模型
    python main.py predict           # 进行预测
    python main.py chart             # 计算星盘
"""

import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import structlog
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

# 内部模块
from h_pulse.utils.settings import get_settings, BrandColors
from h_pulse.output_generation.api import start_api_server
from h_pulse.prediction_ai import train_model, predict_life_trajectory
from h_pulse.data_collection import calculate_bazi, calculate_ziwei, calculate_natal_chart
from h_pulse.quantum_engine import simulate_life_superposition
from h_pulse.output_generation.report import generate_pdf_report
from h_pulse.utils.crypto_anchor import generate_quantum_fingerprint, sign_data

# 配置日志
logger = structlog.get_logger()

# 创建Typer应用
app = typer.Typer(
    name="H-Pulse",
    help="H-Pulse Quantum Prediction System - 量子生命轨迹预测系统",
    add_completion=True
)

# 创建Rich控制台
console = Console()

# 品牌色彩
COLORS = BrandColors()


@app.callback()
def callback():
    """
    H-Pulse Quantum Prediction System
    
    精准 · 独特 · 不可逆
    """
    pass


@app.command()
def demo(
    name: str = typer.Option("张三", "--name", "-n", help="姓名"),
    gender: str = typer.Option("男", "--gender", "-g", help="性别：男/女"),
    birth_datetime: str = typer.Option("1990-01-01T12:00:00+08:00", "--birth", "-b", help="出生时间(ISO格式)"),
    longitude: float = typer.Option(116.4074, "--lon", help="出生地经度"),
    latitude: float = typer.Option(39.9042, "--lat", help="出生地纬度"),
    timezone: str = typer.Option("Asia/Shanghai", "--tz", help="时区"),
    output_report: bool = typer.Option(True, "--report/--no-report", help="是否生成PDF报告")
):
    """运行预测演示"""
    console.print(Panel.fit(
        f"[bold {COLORS.QUANTUM_BLUE}]H-Pulse Quantum Prediction System[/bold {COLORS.QUANTUM_BLUE}]\n"
        f"[{COLORS.NEURAL_RED}]正在为 {name} 进行生命轨迹预测...[/{COLORS.NEURAL_RED}]",
        title="演示模式",
        border_style=COLORS.MYSTIC_PURPLE
    ))
    
    # 构建出生数据
    birth_data = {
        'name': name,
        'gender': gender,
        'birth_datetime': birth_datetime,
        'longitude': longitude,
        'latitude': latitude,
        'timezone': timezone,
        'user_id': f"demo_{datetime.now().timestamp()}"
    }
    
    with Progress(
        SpinnerColumn(style=COLORS.QUANTUM_BLUE),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        # 1. 计算四柱八字
        task1 = progress.add_task("计算四柱八字...", total=1)
        try:
            birth_dt = datetime.fromisoformat(birth_datetime)
            bazi_chart = calculate_bazi(birth_dt, longitude, latitude, gender)
            progress.update(task1, completed=1)
            
            # 显示八字
            bazi_table = Table(title="四柱八字", border_style=COLORS.MYSTIC_PURPLE)
            bazi_table.add_column("柱位", style="bold")
            bazi_table.add_column("干支", style=COLORS.PREDICTION_GREEN)
            bazi_table.add_row("年柱", bazi_chart.year.ganzhi)
            bazi_table.add_row("月柱", bazi_chart.month.ganzhi)
            bazi_table.add_row("日柱", bazi_chart.day.ganzhi)
            bazi_table.add_row("时柱", bazi_chart.hour.ganzhi)
            console.print(bazi_table)
            
        except Exception as e:
            console.print(f"[red]八字计算失败：{e}[/red]")
            progress.update(task1, completed=1)
        
        # 2. 计算紫微斗数
        task2 = progress.add_task("计算紫微斗数...", total=1)
        try:
            ziwei_chart = calculate_ziwei(birth_dt, longitude, latitude, gender)
            progress.update(task2, completed=1)
            
            # 显示命宫
            console.print(f"\n[bold]紫微斗数命宫：[/bold] {ziwei_chart.ming_gong.name}")
            console.print(f"主星：{', '.join([s.name for s in ziwei_chart.ming_gong.main_stars])}")
            
        except Exception as e:
            console.print(f"[red]紫微计算失败：{e}[/red]")
            progress.update(task2, completed=1)
        
        # 3. 计算西方星盘
        task3 = progress.add_task("计算西方星盘...", total=1)
        try:
            natal_chart = calculate_natal_chart(birth_dt, longitude, latitude)
            progress.update(task3, completed=1)
            
            # 显示主要行星
            planet_table = Table(title="行星位置", border_style=COLORS.MYSTIC_PURPLE)
            planet_table.add_column("行星", style="bold")
            planet_table.add_column("星座", style=COLORS.QUANTUM_BLUE)
            planet_table.add_column("度数", style=COLORS.NEURAL_RED)
            
            for planet in natal_chart.planets[:5]:  # 显示前5个行星
                planet_table.add_row(
                    planet.name,
                    planet.sign,
                    f"{planet.degree:.2f}°"
                )
            console.print(planet_table)
            
        except Exception as e:
            console.print(f"[red]星盘计算失败：{e}[/red]")
            progress.update(task3, completed=1)
        
        # 4. 量子模拟
        task4 = progress.add_task("执行量子生命模拟...", total=1)
        try:
            # 示例事件
            possible_events = [
                {"type": "career", "description": "职业突破机会", "timestamp": "2025-06-15T10:00:00"},
                {"type": "relationship", "description": "重要感情相遇", "timestamp": "2025-08-20T14:00:00"},
                {"type": "health", "description": "健康调整期", "timestamp": "2025-10-01T08:00:00"},
                {"type": "wealth", "description": "财富增长机遇", "timestamp": "2026-01-15T16:00:00"},
            ]
            
            quantum_state = simulate_life_superposition(
                birth_data,
                {"bazi": bazi_chart.to_dict() if 'bazi_chart' in locals() else {}},
                possible_events
            )
            progress.update(task4, completed=1)
            
            console.print(f"\n[bold]量子指纹：[/bold] {quantum_state.quantum_fingerprint[:32]}...")
            console.print(f"[bold]叠加态维度：[/bold] {len(quantum_state.events)}")
            console.print(f"[bold]纠缠熵：[/bold] {quantum_state.entanglement_entropy:.4f}")
            
        except Exception as e:
            console.print(f"[red]量子模拟失败：{e}[/red]")
            progress.update(task4, completed=1)
        
        # 5. AI预测
        task5 = progress.add_task("AI深度预测分析...", total=1)
        try:
            prediction_result = predict_life_trajectory(birth_data)
            progress.update(task5, completed=1)
            
            # 显示预测结果
            console.print(Panel.fit(
                f"[bold {COLORS.PREDICTION_GREEN}]预测完成！[/bold {COLORS.PREDICTION_GREEN}]",
                title="预测结果",
                border_style=COLORS.MYSTIC_PURPLE
            ))
            
            # 显示重要事件
            event_table = Table(title="生命轨迹重要事件", border_style=COLORS.MYSTIC_PURPLE)
            event_table.add_column("时间", style="bold")
            event_table.add_column("类型", style=COLORS.QUANTUM_BLUE)
            event_table.add_column("描述", style=COLORS.NEURAL_RED)
            event_table.add_column("概率", style=COLORS.PREDICTION_GREEN)
            
            for event in prediction_result['life_trajectory']['events'][:5]:
                event_table.add_row(
                    event['expected_date'][:10],
                    event['type'],
                    event['description'][:30] + "...",
                    f"{event['probability']:.1%}"
                )
            console.print(event_table)
            
            # 显示整体趋势
            console.print(f"\n[bold]整体趋势：[/bold]")
            for key, value in prediction_result['life_trajectory']['overall_trend'].items():
                console.print(f"  • {key}: {value}")
            
        except Exception as e:
            console.print(f"[red]AI预测失败：{e}[/red]")
            progress.update(task5, completed=1)
            prediction_result = None
    
    # 6. 生成报告
    if output_report and prediction_result:
        with Progress(
            SpinnerColumn(style=COLORS.QUANTUM_BLUE),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task6 = progress.add_task("生成PDF报告...", total=1)
            try:
                report_filename = f"demo_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                report_path = generate_pdf_report(prediction_result, report_filename)
                progress.update(task6, completed=1)
                
                console.print(f"\n[bold green]报告已生成：[/bold green] {report_path}")
                
            except Exception as e:
                console.print(f"[red]报告生成失败：{e}[/red]")
                progress.update(task6, completed=1)
    
    console.print("\n[bold]演示完成！[/bold]")


@app.command()
def api(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="API服务器地址"),
    port: int = typer.Option(8000, "--port", "-p", help="API服务器端口"),
    reload: bool = typer.Option(False, "--reload", "-r", help="开启热重载（开发模式）")
):
    """启动API服务器"""
    console.print(Panel.fit(
        f"[bold {COLORS.QUANTUM_BLUE}]启动H-Pulse API服务器[/bold {COLORS.QUANTUM_BLUE}]\n"
        f"地址: http://{host}:{port}\n"
        f"文档: http://{host}:{port}/docs\n"
        f"GraphQL: http://{host}:{port}/graphql",
        title="API模式",
        border_style=COLORS.MYSTIC_PURPLE
    ))
    
    settings = get_settings()
    settings.debug = reload
    
    try:
        start_api_server(host=host, port=port)
    except KeyboardInterrupt:
        console.print("\n[yellow]API服务器已停止[/yellow]")
    except Exception as e:
        console.print(f"[red]API服务器错误：{e}[/red]")
        raise typer.Exit(1)


@app.command()
def train(
    epochs: int = typer.Option(10, "--epochs", "-e", help="训练轮数"),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="批次大小"),
    learning_rate: float = typer.Option(0.001, "--lr", help="学习率"),
    output_dir: str = typer.Option("./models", "--output", "-o", help="模型保存目录")
):
    """训练AI预测模型"""
    console.print(Panel.fit(
        f"[bold {COLORS.QUANTUM_BLUE}]训练AI预测模型[/bold {COLORS.QUANTUM_BLUE}]",
        title="训练模式",
        border_style=COLORS.MYSTIC_PURPLE
    ))
    
    config = {
        'num_epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'model_dir': Path(output_dir)
    }
    
    console.print(f"训练配置：")
    console.print(f"  • 训练轮数：{epochs}")
    console.print(f"  • 批次大小：{batch_size}")
    console.print(f"  • 学习率：{learning_rate}")
    console.print(f"  • 输出目录：{output_dir}")
    
    try:
        with Progress(
            SpinnerColumn(style=COLORS.QUANTUM_BLUE),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"训练中...", total=epochs)
            
            # 模拟训练进度
            import time
            for i in range(epochs):
                time.sleep(0.5)  # 实际训练会更久
                progress.update(task, advance=1, description=f"训练中... Epoch {i+1}/{epochs}")
            
            # 调用实际训练函数
            train_model(config)
            
        console.print(f"\n[bold green]训练完成！[/bold green]")
        console.print(f"模型已保存到：{output_dir}")
        
    except Exception as e:
        console.print(f"[red]训练失败：{e}[/red]")
        raise typer.Exit(1)


@app.command()
def predict(
    input_file: Optional[Path] = typer.Option(None, "--input", "-i", help="输入数据JSON文件"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="输出结果JSON文件"),
    report: bool = typer.Option(False, "--report", "-r", help="生成PDF报告")
):
    """执行生命轨迹预测"""
    console.print(Panel.fit(
        f"[bold {COLORS.QUANTUM_BLUE}]执行生命轨迹预测[/bold {COLORS.QUANTUM_BLUE}]",
        title="预测模式",
        border_style=COLORS.MYSTIC_PURPLE
    ))
    
    # 读取输入数据
    if input_file:
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                birth_data = json.load(f)
            console.print(f"[green]已加载输入数据：{input_file}[/green]")
        except Exception as e:
            console.print(f"[red]读取输入文件失败：{e}[/red]")
            raise typer.Exit(1)
    else:
        # 交互式输入
        console.print("[yellow]请输入预测信息：[/yellow]")
        birth_data = {
            'name': typer.prompt("姓名", default="匿名"),
            'gender': typer.prompt("性别(男/女)", default="男"),
            'birth_datetime': typer.prompt("出生时间(ISO格式)", default="1990-01-01T12:00:00+08:00"),
            'longitude': float(typer.prompt("出生地经度", default="116.4074")),
            'latitude': float(typer.prompt("出生地纬度", default="39.9042")),
            'timezone': typer.prompt("时区", default="Asia/Shanghai"),
            'user_id': f"predict_{datetime.now().timestamp()}"
        }
    
    # 执行预测
    try:
        with console.status("[bold green]预测中...", spinner="dots"):
            prediction_result = predict_life_trajectory(birth_data)
        
        console.print("[bold green]预测完成！[/bold green]")
        
        # 显示结果摘要
        console.print(f"\n[bold]预测摘要：[/bold]")
        console.print(f"用户ID: {prediction_result['user_id']}")
        console.print(f"预测时间: {prediction_result['prediction_time']}")
        console.print(f"置信度: {prediction_result['confidence_metrics']['overall_confidence']:.2%}")
        console.print(f"事件数量: {len(prediction_result['life_trajectory']['events'])}")
        
        # 保存结果
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(prediction_result, f, ensure_ascii=False, indent=2)
            console.print(f"\n[green]结果已保存到：{output_file}[/green]")
        
        # 生成报告
        if report:
            report_filename = f"prediction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            report_path = generate_pdf_report(prediction_result, report_filename)
            console.print(f"[green]报告已生成：{report_path}[/green]")
        
    except Exception as e:
        console.print(f"[red]预测失败：{e}[/red]")
        raise typer.Exit(1)


@app.command()
def chart(
    chart_type: str = typer.Argument(..., help="星盘类型：bazi/ziwei/natal"),
    birth_datetime: str = typer.Option(..., "--birth", "-b", help="出生时间(ISO格式)"),
    longitude: float = typer.Option(..., "--lon", help="出生地经度"),
    latitude: float = typer.Option(..., "--lat", help="出生地纬度"),
    gender: str = typer.Option("男", "--gender", "-g", help="性别：男/女"),
    output_format: str = typer.Option("json", "--format", "-f", help="输出格式：json/table")
):
    """计算并显示星盘"""
    console.print(Panel.fit(
        f"[bold {COLORS.QUANTUM_BLUE}]计算{chart_type}星盘[/bold {COLORS.QUANTUM_BLUE}]",
        title="星盘模式",
        border_style=COLORS.MYSTIC_PURPLE
    ))
    
    try:
        birth_dt = datetime.fromisoformat(birth_datetime)
        
        if chart_type == "bazi":
            chart = calculate_bazi(birth_dt, longitude, latitude, gender)
            if output_format == "table":
                table = Table(title="四柱八字", border_style=COLORS.MYSTIC_PURPLE)
                table.add_column("柱位", style="bold")
                table.add_column("干支", style=COLORS.PREDICTION_GREEN)
                table.add_column("五行", style=COLORS.QUANTUM_BLUE)
                table.add_row("年柱", chart.year.ganzhi, f"{chart.year.gan_wuxing}+{chart.year.zhi_wuxing}")
                table.add_row("月柱", chart.month.ganzhi, f"{chart.month.gan_wuxing}+{chart.month.zhi_wuxing}")
                table.add_row("日柱", chart.day.ganzhi, f"{chart.day.gan_wuxing}+{chart.day.zhi_wuxing}")
                table.add_row("时柱", chart.hour.ganzhi, f"{chart.hour.gan_wuxing}+{chart.hour.zhi_wuxing}")
                console.print(table)
            else:
                console.print_json(data=chart.to_dict())
                
        elif chart_type == "ziwei":
            chart = calculate_ziwei(birth_dt, longitude, latitude, gender)
            if output_format == "table":
                table = Table(title="紫微斗数十二宫", border_style=COLORS.MYSTIC_PURPLE)
                table.add_column("宫位", style="bold")
                table.add_column("主星", style=COLORS.PREDICTION_GREEN)
                for gong in chart.palaces[:6]:  # 显示前6宫
                    stars = ", ".join([s.name for s in gong.main_stars])
                    table.add_row(gong.name, stars)
                console.print(table)
            else:
                console.print_json(data=chart.to_dict())
                
        elif chart_type == "natal":
            chart = calculate_natal_chart(birth_dt, longitude, latitude)
            if output_format == "table":
                table = Table(title="西方星盘", border_style=COLORS.MYSTIC_PURPLE)
                table.add_column("行星", style="bold")
                table.add_column("星座", style=COLORS.PREDICTION_GREEN)
                table.add_column("度数", style=COLORS.QUANTUM_BLUE)
                table.add_column("宫位", style=COLORS.NEURAL_RED)
                for planet in chart.planets:
                    table.add_row(
                        planet.name,
                        planet.sign,
                        f"{planet.degree:.2f}°",
                        str(planet.house)
                    )
                console.print(table)
            else:
                console.print_json(data=chart.to_dict())
        else:
            console.print(f"[red]未知的星盘类型：{chart_type}[/red]")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"[red]星盘计算失败：{e}[/red]")
        raise typer.Exit(1)


@app.command()
def version():
    """显示版本信息"""
    settings = get_settings()
    
    console.print(Panel.fit(
        f"[bold {COLORS.QUANTUM_BLUE}]H-Pulse Quantum Prediction System[/bold {COLORS.QUANTUM_BLUE}]\n"
        f"版本: {settings.project_version}\n"
        f"[{COLORS.NEURAL_RED}]精准 · 独特 · 不可逆[/{COLORS.NEURAL_RED}]",
        title="版本信息",
        border_style=COLORS.MYSTIC_PURPLE
    ))
    
    # 显示系统信息
    info_table = Table(show_header=False, border_style=COLORS.MYSTIC_PURPLE)
    info_table.add_column("属性", style="bold")
    info_table.add_column("值", style=COLORS.PREDICTION_GREEN)
    
    info_table.add_row("项目名称", settings.project_name)
    info_table.add_row("版本", settings.project_version)
    info_table.add_row("API地址", f"{settings.api_host}:{settings.api_port}")
    info_table.add_row("量子后端", settings.quantum_backend)
    info_table.add_row("区块链", "已启用" if settings.blockchain_enabled else "未启用")
    info_table.add_row("日志级别", settings.log_level)
    
    console.print(info_table)


def main():
    """主函数"""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]程序已中断[/yellow]")
        sys.exit(0)
    except Exception as e:
        logger.error("程序异常", error=str(e))
        console.print(f"\n[red]程序异常：{e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()