"""
模型训练模块
实现模型训练、验证、保存等功能
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import structlog
from datetime import datetime
from tqdm import tqdm

from .model import LifeTrajectoryModel
from ..utils.settings import get_settings

logger = structlog.get_logger()


class LifeTrajectoryDataset(Dataset):
    """生命轨迹数据集"""
    
    def __init__(self, data_path: Path, mode: str = 'train'):
        """
        Args:
            data_path: 数据路径
            mode: 'train', 'val', 或 'test'
        """
        self.data_path = data_path
        self.mode = mode
        self.samples = []
        
        # 加载数据
        self._load_data()
    
    def _load_data(self):
        """加载数据（这里使用合成数据作为示例）"""
        # 实际应该从文件加载
        # 这里生成一些示例数据
        n_samples = 1000 if self.mode == 'train' else 200
        
        for i in range(n_samples):
            sample = {
                'id': f"{self.mode}_{i}",
                'astro_features': np.random.randn(10, 64).astype(np.float32),
                'eastern_features': np.random.randn(10, 128).astype(np.float32),
                'western_features': np.random.randn(10, 128).astype(np.float32),
                'quantum_features': np.random.randn(10, 256).astype(np.float32),
                'target_events': np.random.randint(0, 256, size=5),
                'target_times': np.random.randint(0, 365, size=5),
                'metadata': {
                    'birth_date': '1990-01-01',
                    'location': {'lat': 39.9, 'lon': 116.4}
                }
            }
            self.samples.append(sample)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        # 转换为张量
        item = {
            'astro_features': torch.from_numpy(sample['astro_features']),
            'eastern_features': torch.from_numpy(sample['eastern_features']),
            'western_features': torch.from_numpy(sample['western_features']),
            'quantum_features': torch.from_numpy(sample['quantum_features']),
            'target_events': torch.tensor(sample['target_events'], dtype=torch.long),
            'target_times': torch.tensor(sample['target_times'], dtype=torch.long),
            'attention_mask': torch.ones(10, dtype=torch.float32)
        }
        
        return item


class Trainer:
    """模型训练器"""
    
    def __init__(self, model: LifeTrajectoryModel, config: Optional[Dict[str, Any]] = None):
        self.model = model
        self.settings = get_settings()
        
        # 训练配置
        self.config = config or {
            'learning_rate': 1e-4,
            'batch_size': 32,
            'num_epochs': 100,
            'warmup_steps': 1000,
            'gradient_clip': 1.0,
            'save_steps': 1000,
            'eval_steps': 500,
            'log_steps': 100,
            'device': self.settings.model_device
        }
        
        # 设备
        self.device = torch.device(self.config['device'])
        self.model.to(self.device)
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=0.01
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['num_epochs']
        )
        
        # 损失函数
        self.event_criterion = nn.CrossEntropyLoss()
        self.time_criterion = nn.CrossEntropyLoss()
        self.confidence_criterion = nn.BCELoss()
        
        # 训练状态
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.training_history = []
    
    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader,
             save_dir: Path):
        """
        训练模型
        
        Args:
            train_dataloader: 训练数据加载器
            val_dataloader: 验证数据加载器
            save_dir: 模型保存目录
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("开始训练", 
                   num_epochs=self.config['num_epochs'],
                   batch_size=self.config['batch_size'],
                   device=self.device)
        
        for epoch in range(self.config['num_epochs']):
            # 训练阶段
            train_loss = self._train_epoch(train_dataloader, epoch)
            
            # 验证阶段
            val_loss = self._validate(val_dataloader)
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录历史
            self.training_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            })
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(save_dir / 'best_model.pth', epoch, val_loss)
            
            # 定期保存
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(save_dir / f'model_epoch_{epoch}.pth', epoch, val_loss)
            
            logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        # 保存训练历史
        with open(save_dir / 'training_history.json', 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        logger.info("训练完成", best_val_loss=self.best_val_loss)
    
    def _train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            # 移动到设备
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # 前向传播
            outputs = self.model(batch)
            
            # 计算损失
            loss = self._compute_loss(outputs, batch)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config['gradient_clip']
            )
            
            # 更新参数
            self.optimizer.step()
            
            # 记录
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # 更新进度条
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            # 定期记录
            if self.global_step % self.config['log_steps'] == 0:
                logger.debug("训练步骤", 
                           step=self.global_step,
                           loss=loss.item(),
                           lr=self.optimizer.param_groups[0]['lr'])
        
        return total_loss / num_batches
    
    def _validate(self, dataloader: DataLoader) -> float:
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                # 移动到设备
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # 前向传播
                outputs = self.model(batch)
                
                # 计算损失
                loss = self._compute_loss(outputs, batch)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def _compute_loss(self, outputs: Dict[str, torch.Tensor], 
                     batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算损失"""
        # 事件预测损失
        event_loss = 0.0
        if 'target_events' in batch:
            # 假设目标是多个事件，取第一个作为主要事件
            target_event = batch['target_events'][:, 0]
            event_loss = self.event_criterion(outputs['event_logits'], target_event)
        
        # 时间预测损失
        time_loss = 0.0
        if 'target_times' in batch:
            target_time = batch['target_times'][:, 0]
            time_loss = self.time_criterion(outputs['time_logits'], target_time)
        
        # 置信度损失（这里使用合成标签）
        confidence_target = torch.ones_like(outputs['confidence']) * 0.8
        confidence_loss = self.confidence_criterion(outputs['confidence'], confidence_target)
        
        # 不确定性损失（负对数似然）
        uncertainty_loss = 0.5 * (outputs['uncertainty_logvar'] + 
                                 torch.exp(-outputs['uncertainty_logvar'])).mean()
        
        # 总损失
        total_loss = (event_loss + time_loss + 
                     0.1 * confidence_loss + 
                     0.01 * uncertainty_loss)
        
        return total_loss
    
    def _save_checkpoint(self, path: Path, epoch: int, val_loss: float):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
            'global_step': self.global_step
        }
        
        torch.save(checkpoint, path)
        logger.info(f"模型已保存: {path}")
    
    def load_checkpoint(self, path: Path):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)
        
        logger.info(f"模型已加载: {path}")
        
        return checkpoint['epoch'], checkpoint['val_loss']


def train_model(data_path: Path, save_dir: Path, config: Optional[Dict[str, Any]] = None):
    """训练模型的便捷函数"""
    # 创建数据集
    train_dataset = LifeTrajectoryDataset(data_path, mode='train')
    val_dataset = LifeTrajectoryDataset(data_path, mode='val')
    
    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 32) if config else 32,
        shuffle=True,
        num_workers=4
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 32) if config else 32,
        shuffle=False,
        num_workers=4
    )
    
    # 创建模型
    model = LifeTrajectoryModel()
    
    # 创建训练器
    trainer = Trainer(model, config)
    
    # 开始训练
    trainer.train(train_dataloader, val_dataloader, save_dir)
    
    return trainer


if __name__ == "__main__":
    # 测试训练
    from pathlib import Path
    
    data_path = Path("./data")  # 实际应该指向真实数据
    save_dir = Path("./models")
    
    # 训练配置
    train_config = {
        'learning_rate': 1e-4,
        'batch_size': 16,
        'num_epochs': 10,
        'device': 'cpu'  # 测试用CPU
    }
    
    # 开始训练
    trainer = train_model(data_path, save_dir, train_config)