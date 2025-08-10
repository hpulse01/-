"""
H-Pulse AI预测模型
实现Transformer+GNN融合网络，用于生命轨迹预测
"""

from .model import *
from .train import *
from .infer import *

__all__ = [
    # model
    'LifeTrajectoryModel',
    'TransformerEncoder',
    'GraphNeuralNetwork',
    'MultiModalFusion',
    
    # train
    'Trainer',
    'train_model',
    
    # infer
    'Predictor',
    'predict_life_trajectory',
]