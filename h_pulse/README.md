# H-Pulse Quantum Prediction System

高精度生命轨迹预测系统（东方/西方天文命理 × 量子仿真 × AI）。

- 语言：中文为主
- 品牌配色：量子蓝#00F5FF、神经红#FF6B6B、神秘紫#764BA2、预测绿#4ECDC4、深空黑#0A0A1A

## 安装

1) Python 3.10+
2) 创建并激活虚拟环境
3) 安装依赖：

```bash
pip install -r requirements.txt
```

注：若 `torch/torch-geometric/qiskit/pennylane` 未安装，本项目提供降级纯 NumPy 路径，单元测试仍可通过。

## 运行演示

```bash
python -m h_pulse.main demo
```

将完成：数据特征 → 量子仿真 → AI 推理 → 时间轴 JSON 与图表/PDF → 本地签名与“链上”锚定（本地模拟，标注 synthetic）。

## 启动 API（REST + GraphQL）

```bash
python -m h_pulse.main serve --host 0.0.0.0 --port 8000
# 浏览器打开：http://localhost:8000/docs
# GraphQL Playground: http://localhost:8000/graphql
```

## 训练

```bash
python -m h_pulse.main train --epochs 2 --seed 42
```

## 测试

```bash
pytest -q
```

## 目录

见 `h_pulse/` 包内：
- `utils/` 时间天文、密码学与锚定、配置
- `data_collection/` 东方与西方特征
- `quantum_engine/` 量子模拟器/量子指纹
- `prediction_ai/` 模型、训练、推理
- `output_generation/` 报告、API
- `tests/` 单元/集成测试

## 免责声明

本项目用于技术研究与工程实践示范。预测结果包含不确定性，报告中附置信区间与假设说明。