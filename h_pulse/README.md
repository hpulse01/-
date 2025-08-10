# H-Pulse 量子预测系统
## Quantum Prediction System

> **唯一使命**: 真实、完整、完全、完美地预测某人未来点点滴滴的生命轨迹。  
> **Precision · Uniqueness · Irreversibility**

## 系统架构

H-Pulse 是一个融合东西方命理学、量子计算和深度学习的生命轨迹预测系统。通过高精度天文计算、量子叠加态模拟和 Transformer+GNN 融合网络，实现对个体生命轨迹的精准预测。

### 核心特性

- **双体系融合**: 同时实现东方（四柱八字、紫微斗数）和西方（行星占星）两套高精度体系
- **量子计算**: 基于量子叠加态和纠缠模拟生命可能性空间
- **AI预测**: Transformer + Graph Neural Network 多模态融合
- **区块链锚定**: Ed25519签名 + Web3链上锚定确保预测不可篡改
- **高精度天文**: 考虑岁差、章动、光行差，支持真太阳时计算

## 快速开始

### 环境要求
- Python >= 3.10
- CUDA (可选，用于GPU加速)

### 安装

```bash
# 克隆项目
cd /workspace/h_pulse

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 安装项目
pip install -e .
```

### 运行演示

```bash
# 运行完整演示
python -m h_pulse.main demo

# 启动API服务
python -m h_pulse.main serve

# 训练模型（需要数据集）
python -m h_pulse.main train --data-path ./data
```

### API使用

启动服务后，访问以下端点：

- API文档: http://localhost:8000/docs
- GraphQL: http://localhost:8000/graphql
- 健康检查: http://localhost:8000/healthz

#### 预测请求示例

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "birth_datetime": "1990-01-01T12:00:00+08:00",
    "birth_location": {
      "latitude": 39.9042,
      "longitude": 116.4074,
      "timezone": "Asia/Shanghai"
    },
    "name": "测试用户"
  }'
```

## 项目结构

```
h_pulse/
├── data_collection/      # 数据采集模块
│   ├── eastern.py       # 东方命理：四柱八字、紫微斗数
│   └── western.py       # 西方占星：行星、宫位、相位
├── data_processing/     # 数据处理管道
├── quantum_engine/      # 量子计算引擎
│   └── simulator.py     # 量子叠加态模拟、量子指纹
├── prediction_ai/       # AI预测模型
│   ├── model.py        # Transformer+GNN架构
│   ├── train.py        # 训练脚本
│   └── infer.py        # 推理接口
├── output_generation/   # 输出生成
│   ├── report.py       # PDF报告、可视化
│   └── api.py          # FastAPI + GraphQL
├── utils/              # 工具模块
│   ├── time_astro.py   # 时间天文计算
│   ├── crypto_anchor.py # 加密签名与链上锚定
│   └── settings.py     # 配置管理
└── tests/              # 测试用例
```

## 技术规范

### 时间与空间基准
- 时标：UTC、TT/TDB，考虑ΔT修正
- 天文：JPL DE430/DE440星历，处理岁差/章动/光行差
- 坐标：WGS84大地测量系统

### 品牌视觉
- 量子蓝 #00F5FF
- 神经红 #FF6B6B  
- 神秘紫 #764BA2
- 预测绿 #4ECDC4
- 深空黑 #0A0A1A

## 测试

```bash
# 运行所有测试
pytest

# 运行特定模块测试
pytest tests/test_time_astro.py

# 生成覆盖率报告
pytest --cov=h_pulse --cov-report=html
```

## 许可证

Copyright © 2025 H-Pulse Quantum Prediction System. All rights reserved.