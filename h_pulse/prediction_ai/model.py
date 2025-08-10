"""
生命轨迹预测模型
Transformer + Graph Neural Network 多模态融合架构

模型架构：
1. 特征编码器：处理天文/命理/环境特征
2. Transformer编码器：捕获时序依赖
3. 图神经网络：建模事件间因果关系
4. 多模态融合：整合所有信息
5. 预测头：输出事件概率和时间分布
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import math
import structlog

from ..utils.settings import get_settings

logger = structlog.get_logger()


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, 
               value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # 线性变换并分头
        Q = self.W_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        context = torch.matmul(attn_weights, V)
        
        # 合并多头
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        output = self.W_o(context)
        
        return output


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 自注意力
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class TransformerEncoder(nn.Module):
    """Transformer编码器"""
    
    def __init__(self, d_model: int, n_heads: int, n_layers: int, 
                d_ff: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 添加位置编码
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # 通过编码器层
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.norm(x)
        
        return x


class GraphNeuralNetwork(nn.Module):
    """图神经网络（用于事件关系建模）"""
    
    def __init__(self, in_features: int, hidden_features: int, 
                out_features: int, n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # 第一层
        self.convs.append(GATConv(in_features, hidden_features, heads=4, concat=True))
        self.bns.append(nn.BatchNorm1d(hidden_features * 4))
        
        # 中间层
        for _ in range(n_layers - 2):
            self.convs.append(GATConv(hidden_features * 4, hidden_features, heads=4, concat=True))
            self.bns.append(nn.BatchNorm1d(hidden_features * 4))
        
        # 最后一层
        self.convs.append(GATConv(hidden_features * 4, out_features, heads=1, concat=False))
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ELU()
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
               batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        for i, (conv, bn) in enumerate(zip(self.convs[:-1], self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = self.activation(x)
            x = self.dropout(x)
        
        # 最后一层
        x = self.convs[-1](x, edge_index)
        
        # 全局池化
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        return x


class MultiModalFusion(nn.Module):
    """多模态融合模块"""
    
    def __init__(self, astro_dim: int, eastern_dim: int, western_dim: int,
                quantum_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        # 各模态的投影层
        self.astro_proj = nn.Linear(astro_dim, hidden_dim)
        self.eastern_proj = nn.Linear(eastern_dim, hidden_dim)
        self.western_proj = nn.Linear(western_dim, hidden_dim)
        self.quantum_proj = nn.Linear(quantum_dim, hidden_dim)
        
        # 跨模态注意力
        self.cross_attn = MultiHeadAttention(hidden_dim, n_heads=8, dropout=dropout)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 4, 4),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, astro_feat: torch.Tensor, eastern_feat: torch.Tensor,
               western_feat: torch.Tensor, quantum_feat: torch.Tensor) -> torch.Tensor:
        # 投影到同一空间
        astro_h = self.astro_proj(astro_feat)
        eastern_h = self.eastern_proj(eastern_feat)
        western_h = self.western_proj(western_feat)
        quantum_h = self.quantum_proj(quantum_feat)
        
        # 堆叠特征
        features = torch.stack([astro_h, eastern_h, western_h, quantum_h], dim=1)
        
        # 跨模态注意力
        attended = self.cross_attn(features, features, features)
        
        # 拼接所有特征
        concat_feat = torch.cat([
            attended[:, 0], attended[:, 1], 
            attended[:, 2], attended[:, 3]
        ], dim=-1)
        
        # 计算门控权重
        gates = self.gate(concat_feat).unsqueeze(-1)
        
        # 加权融合
        weighted_features = features * gates
        fused = weighted_features.sum(dim=1)
        
        # 最终融合
        output = self.fusion(concat_feat) + fused
        
        return output


class LifeTrajectoryModel(nn.Module):
    """生命轨迹预测主模型"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        if config is None:
            settings = get_settings()
            config = {
                'hidden_size': settings.model_hidden_size,
                'num_heads': settings.model_num_heads,
                'num_layers': settings.model_num_layers,
                'dropout': settings.model_dropout,
                'max_seq_length': settings.model_max_sequence_length
            }
        
        self.config = config
        hidden_size = config['hidden_size']
        
        # 特征维度
        self.astro_dim = 64      # 天文特征
        self.eastern_dim = 128   # 东方命理特征
        self.western_dim = 128   # 西方占星特征  
        self.quantum_dim = 256   # 量子态特征
        
        # 特征编码器
        self.astro_encoder = nn.Sequential(
            nn.Linear(self.astro_dim, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size)
        )
        
        self.eastern_encoder = nn.Sequential(
            nn.Linear(self.eastern_dim, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size)
        )
        
        self.western_encoder = nn.Sequential(
            nn.Linear(self.western_dim, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size)
        )
        
        self.quantum_encoder = nn.Sequential(
            nn.Linear(self.quantum_dim, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size)
        )
        
        # Transformer编码器
        self.transformer = TransformerEncoder(
            d_model=hidden_size,
            n_heads=config['num_heads'],
            n_layers=config['num_layers'],
            d_ff=hidden_size * 4,
            max_len=config['max_seq_length'],
            dropout=config['dropout']
        )
        
        # 图神经网络
        self.gnn = GraphNeuralNetwork(
            in_features=hidden_size,
            hidden_features=hidden_size,
            out_features=hidden_size,
            n_layers=3,
            dropout=config['dropout']
        )
        
        # 多模态融合
        self.fusion = MultiModalFusion(
            astro_dim=hidden_size,
            eastern_dim=hidden_size,
            western_dim=hidden_size,
            quantum_dim=hidden_size,
            hidden_dim=hidden_size,
            dropout=config['dropout']
        )
        
        # 预测头
        self.event_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(hidden_size * 2, 256)  # 256种事件类型
        )
        
        self.time_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(hidden_size, 365)  # 365天时间分布
        )
        
        self.confidence_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # 不确定性估计
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 2)  # 均值和方差
        )
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            batch: 包含各种特征的字典
                - astro_features: [batch_size, seq_len, astro_dim]
                - eastern_features: [batch_size, seq_len, eastern_dim]
                - western_features: [batch_size, seq_len, western_dim]
                - quantum_features: [batch_size, seq_len, quantum_dim]
                - event_graph: Data对象，包含edge_index等
                - attention_mask: [batch_size, seq_len]
        
        Returns:
            预测结果字典
        """
        # 获取输入
        astro_feat = batch['astro_features']
        eastern_feat = batch['eastern_features']
        western_feat = batch['western_features']
        quantum_feat = batch['quantum_features']
        mask = batch.get('attention_mask', None)
        
        batch_size, seq_len = astro_feat.shape[:2]
        
        # 特征编码
        astro_encoded = self.astro_encoder(astro_feat)
        eastern_encoded = self.eastern_encoder(eastern_feat)
        western_encoded = self.western_encoder(western_feat)
        quantum_encoded = self.quantum_encoder(quantum_feat)
        
        # 合并序列特征
        seq_features = astro_encoded + eastern_encoded + western_encoded + quantum_encoded
        
        # Transformer编码
        transformer_out = self.transformer(seq_features, mask)
        
        # 图神经网络（如果有事件图）
        if 'event_graph' in batch and batch['event_graph'] is not None:
            graph_data = batch['event_graph']
            
            # 将序列特征转换为节点特征
            node_features = transformer_out.view(-1, transformer_out.size(-1))
            
            # GNN处理
            graph_out = self.gnn(
                node_features, 
                graph_data.edge_index,
                graph_data.batch if hasattr(graph_data, 'batch') else None
            )
            
            # 重塑回批次形式
            if graph_out.dim() == 2 and graph_out.size(0) == batch_size:
                graph_features = graph_out.unsqueeze(1).expand(-1, seq_len, -1)
            else:
                graph_features = graph_out.view(batch_size, seq_len, -1)
        else:
            graph_features = transformer_out
        
        # 多模态融合
        # 取序列的平均作为全局表示
        astro_global = astro_encoded.mean(dim=1)
        eastern_global = eastern_encoded.mean(dim=1)
        western_global = western_encoded.mean(dim=1)
        quantum_global = quantum_encoded.mean(dim=1)
        
        fused_features = self.fusion(
            astro_global, eastern_global, 
            western_global, quantum_global
        )
        
        # 预测
        event_logits = self.event_predictor(fused_features)
        time_logits = self.time_predictor(fused_features)
        confidence = self.confidence_predictor(fused_features)
        
        # 不确定性估计
        uncertainty_params = self.uncertainty_head(fused_features)
        uncertainty_mean = uncertainty_params[:, 0]
        uncertainty_logvar = uncertainty_params[:, 1]
        
        # 计算特征贡献度（用于解释性）
        feature_importance = self._compute_feature_importance(
            astro_global, eastern_global, western_global, quantum_global
        )
        
        outputs = {
            'event_logits': event_logits,           # [batch_size, num_events]
            'time_logits': time_logits,             # [batch_size, 365]
            'confidence': confidence,                # [batch_size, 1]
            'uncertainty_mean': uncertainty_mean,    # [batch_size]
            'uncertainty_logvar': uncertainty_logvar,# [batch_size]
            'feature_importance': feature_importance,# [batch_size, 4]
            'hidden_states': fused_features         # [batch_size, hidden_size]
        }
        
        return outputs
    
    def _compute_feature_importance(self, astro: torch.Tensor, eastern: torch.Tensor,
                                  western: torch.Tensor, quantum: torch.Tensor) -> torch.Tensor:
        """计算各模态特征的重要性"""
        # 使用L2范数作为重要性度量
        astro_imp = torch.norm(astro, dim=-1, keepdim=True)
        eastern_imp = torch.norm(eastern, dim=-1, keepdim=True)
        western_imp = torch.norm(western, dim=-1, keepdim=True)
        quantum_imp = torch.norm(quantum, dim=-1, keepdim=True)
        
        # 拼接并归一化
        importance = torch.cat([astro_imp, eastern_imp, western_imp, quantum_imp], dim=-1)
        importance = F.softmax(importance, dim=-1)
        
        return importance
    
    def predict_trajectory(self, features: Dict[str, Any], 
                         num_events: int = 10) -> Dict[str, Any]:
        """
        预测生命轨迹
        
        Args:
            features: 输入特征
            num_events: 预测的事件数量
            
        Returns:
            轨迹预测结果
        """
        self.eval()
        
        with torch.no_grad():
            # 准备批次数据
            batch = self._prepare_batch(features)
            
            # 前向传播
            outputs = self.forward(batch)
            
            # 解析预测结果
            event_probs = F.softmax(outputs['event_logits'], dim=-1)
            time_probs = F.softmax(outputs['time_logits'], dim=-1)
            
            # 选择Top-K事件
            top_k_probs, top_k_indices = torch.topk(event_probs, k=num_events, dim=-1)
            
            # 生成轨迹
            trajectory = []
            for i in range(num_events):
                event_idx = top_k_indices[0, i].item()
                event_prob = top_k_probs[0, i].item()
                
                # 时间分布
                time_dist = time_probs[0].cpu().numpy()
                expected_day = np.argmax(time_dist)
                
                trajectory.append({
                    'event_id': event_idx,
                    'probability': event_prob,
                    'expected_day': expected_day,
                    'time_distribution': time_dist.tolist(),
                    'confidence': outputs['confidence'][0].item(),
                    'uncertainty': torch.exp(outputs['uncertainty_logvar'][0]).item()
                })
            
            # 特征重要性
            feature_imp = outputs['feature_importance'][0].cpu().numpy()
            
            result = {
                'trajectory': trajectory,
                'feature_importance': {
                    'astro': float(feature_imp[0]),
                    'eastern': float(feature_imp[1]),
                    'western': float(feature_imp[2]),
                    'quantum': float(feature_imp[3])
                }
            }
        
        return result
    
    def _prepare_batch(self, features: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """准备批次数据"""
        # 这里简化处理，实际应该有更复杂的预处理
        device = next(self.parameters()).device
        
        batch = {}
        for key in ['astro_features', 'eastern_features', 'western_features', 'quantum_features']:
            if key in features:
                tensor = torch.tensor(features[key], dtype=torch.float32)
                if tensor.dim() == 2:
                    tensor = tensor.unsqueeze(0)  # 添加batch维度
                batch[key] = tensor.to(device)
        
        # 创建注意力掩码
        seq_len = batch['astro_features'].size(1)
        batch['attention_mask'] = torch.ones(1, seq_len).to(device)
        
        return batch


if __name__ == "__main__":
    # 测试模型
    
    # 创建模型
    model = LifeTrajectoryModel()
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建测试数据
    batch_size = 2
    seq_len = 10
    
    batch = {
        'astro_features': torch.randn(batch_size, seq_len, model.astro_dim),
        'eastern_features': torch.randn(batch_size, seq_len, model.eastern_dim),
        'western_features': torch.randn(batch_size, seq_len, model.western_dim),
        'quantum_features': torch.randn(batch_size, seq_len, model.quantum_dim),
        'attention_mask': torch.ones(batch_size, seq_len)
    }
    
    # 前向传播
    outputs = model(batch)
    
    print("\n输出形状:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    # 测试轨迹预测
    print("\n测试轨迹预测:")
    features = {
        'astro_features': np.random.randn(seq_len, model.astro_dim),
        'eastern_features': np.random.randn(seq_len, model.eastern_dim),
        'western_features': np.random.randn(seq_len, model.western_dim),
        'quantum_features': np.random.randn(seq_len, model.quantum_dim)
    }
    
    trajectory = model.predict_trajectory(features, num_events=5)
    
    print("\n预测轨迹:")
    for i, event in enumerate(trajectory['trajectory']):
        print(f"  事件{i+1}: ID={event['event_id']}, "
              f"概率={event['probability']:.3f}, "
              f"预期天数={event['expected_day']}")
    
    print(f"\n特征重要性: {trajectory['feature_importance']}")