"""
系统配置管理模块
包含全局配置、品牌VI、路径管理等
"""

from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseSettings, Field
import os
import json
import structlog

logger = structlog.get_logger()


class BrandColors:
    """品牌VI色彩系统"""
    QUANTUM_BLUE = "#00F5FF"    # 量子蓝
    NEURAL_RED = "#FF6B6B"      # 神经红
    MYSTIC_PURPLE = "#764BA2"   # 神秘紫
    PREDICTION_GREEN = "#4ECDC4" # 预测绿
    DEEP_SPACE_BLACK = "#0A0A1A" # 深空黑
    
    # 辅助色
    LIGHT_GRAY = "#E0E0E0"
    MEDIUM_GRAY = "#9E9E9E"
    WHITE = "#FFFFFF"
    
    # 语义色
    SUCCESS = PREDICTION_GREEN
    WARNING = "#FFA726"
    ERROR = NEURAL_RED
    INFO = QUANTUM_BLUE


class Settings(BaseSettings):
    """系统配置"""
    
    # 项目基本信息
    project_name: str = "H-Pulse Quantum Prediction System"
    project_version: str = "1.1.0"
    project_tagline: str = "Precision · Uniqueness · Irreversibility"
    
    # API配置
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_prefix: str = "/api/v1"
    api_title: str = "H-Pulse API"
    api_description: str = "量子预测系统API接口"
    
    # 路径配置
    base_dir: Path = Path(__file__).parent.parent
    data_dir: Path = base_dir / "data"
    models_dir: Path = base_dir / "models"
    output_dir: Path = base_dir / "output"
    logs_dir: Path = base_dir / "logs"
    assets_dir: Path = base_dir / "assets"
    
    # 天文数据配置
    ephemeris_path: Optional[str] = Field(default=None, env="EPHEMERIS_PATH")
    use_builtin_ephemeris: bool = True
    ephemeris_de_number: int = 440  # JPL DE440
    
    # 量子计算配置
    quantum_backend: str = Field(default="aer_simulator", env="QUANTUM_BACKEND")
    quantum_shots: int = Field(default=1024, env="QUANTUM_SHOTS")
    quantum_seed: Optional[int] = Field(default=42, env="QUANTUM_SEED")
    quantum_optimization_level: int = 3
    
    # AI模型配置
    model_name: str = "h-pulse-transformer-gnn"
    model_device: str = Field(default="cpu", env="MODEL_DEVICE")  # cpu, cuda, mps
    model_batch_size: int = Field(default=32, env="MODEL_BATCH_SIZE")
    model_max_sequence_length: int = 512
    model_hidden_size: int = 768
    model_num_heads: int = 12
    model_num_layers: int = 6
    model_dropout: float = 0.1
    
    # 区块链配置
    blockchain_enabled: bool = Field(default=False, env="BLOCKCHAIN_ENABLED")
    blockchain_network: str = Field(default="polygon", env="BLOCKCHAIN_NETWORK")
    blockchain_rpc_url: Optional[str] = Field(default=None, env="BLOCKCHAIN_RPC_URL")
    blockchain_contract_address: Optional[str] = Field(default=None, env="CONTRACT_ADDRESS")
    blockchain_private_key: Optional[str] = Field(default=None, env="BLOCKCHAIN_PRIVATE_KEY")
    
    # 安全配置
    enable_signature: bool = True
    signature_algorithm: str = "Ed25519"
    hash_algorithm: str = "SHA3-256"
    
    # 日志配置
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = "json"  # json, console
    log_file_rotation: str = "1 day"
    log_file_retention: str = "30 days"
    
    # 报告配置
    report_language: str = Field(default="zh_CN", env="REPORT_LANGUAGE")
    report_timezone: str = Field(default="Asia/Shanghai", env="REPORT_TIMEZONE")
    report_date_format: str = "%Y年%m月%d日 %H:%M:%S"
    report_include_technical_details: bool = True
    
    # 缓存配置
    enable_cache: bool = True
    cache_ttl: int = 3600  # 秒
    cache_backend: str = "memory"  # memory, redis
    
    # 开发配置
    debug: bool = Field(default=False, env="DEBUG")
    testing: bool = Field(default=False, env="TESTING")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ensure_directories()
        
    def _ensure_directories(self):
        """确保必要的目录存在"""
        for dir_path in [self.data_dir, self.models_dir, self.output_dir, 
                        self.logs_dir, self.assets_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def get_ephemeris_path(self) -> Path:
        """获取星历文件路径"""
        if self.ephemeris_path:
            return Path(self.ephemeris_path)
        return self.data_dir / "ephemeris"
    
    def get_model_path(self, model_name: Optional[str] = None) -> Path:
        """获取模型文件路径"""
        name = model_name or self.model_name
        return self.models_dir / f"{name}.pth"
    
    def get_output_path(self, filename: str, subdir: Optional[str] = None) -> Path:
        """获取输出文件路径"""
        if subdir:
            output_dir = self.output_dir / subdir
            output_dir.mkdir(parents=True, exist_ok=True)
            return output_dir / filename
        return self.output_dir / filename
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（隐藏敏感信息）"""
        data = self.dict()
        # 隐藏敏感信息
        sensitive_keys = ['blockchain_private_key', 'blockchain_rpc_url']
        for key in sensitive_keys:
            if key in data and data[key]:
                data[key] = "***HIDDEN***"
        return data
    
    def save_config(self, path: Optional[Path] = None):
        """保存配置到文件"""
        path = path or self.base_dir / "config.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info("配置已保存", path=str(path))
    
    @classmethod
    def load_config(cls, path: Optional[Path] = None) -> 'Settings':
        """从文件加载配置"""
        path = path or Path.cwd() / "config.json"
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info("配置已加载", path=str(path))
            return cls(**data)
        return cls()


# 全局配置实例
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """获取全局配置实例（单例模式）"""
    global _settings
    if _settings is None:
        _settings = Settings()
        logger.info("初始化系统配置", 
                   version=_settings.project_version,
                   debug=_settings.debug,
                   quantum_backend=_settings.quantum_backend)
    return _settings


def reset_settings():
    """重置配置（主要用于测试）"""
    global _settings
    _settings = None


# 导出常用配置
settings = get_settings()
colors = BrandColors()


if __name__ == "__main__":
    # 测试配置
    config = get_settings()
    print(f"项目: {config.project_name} v{config.project_version}")
    print(f"标语: {config.project_tagline}")
    print(f"API: http://{config.api_host}:{config.api_port}")
    print(f"量子后端: {config.quantum_backend}")
    print(f"模型设备: {config.model_device}")
    print(f"\n品牌色彩:")
    print(f"  量子蓝: {colors.QUANTUM_BLUE}")
    print(f"  神经红: {colors.NEURAL_RED}")
    print(f"  神秘紫: {colors.MYSTIC_PURPLE}")
    print(f"  预测绿: {colors.PREDICTION_GREEN}")
    print(f"  深空黑: {colors.DEEP_SPACE_BLACK}")
    
    # 保存配置示例
    config.save_config()