"""
加密签名与区块链锚定模块
实现量子指纹生成、Ed25519签名、SHA3-256哈希、Web3链上锚定等功能
"""

import hashlib
import json
import secrets
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, Union
from pathlib import Path
import structlog

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.backends import default_backend
import sha3  # pysha3
from web3 import Web3
from eth_account import Account
from eth_account.messages import encode_defunct

from .settings import get_settings

logger = structlog.get_logger()


class QuantumFingerprint:
    """量子指纹生成器"""
    
    @staticmethod
    def generate(bio_features: Dict[str, Any], 
                birth_spacetime: Dict[str, Any],
                device_entropy: Optional[bytes] = None) -> str:
        """
        生成量子指纹
        
        Args:
            bio_features: 生物特征数据（生日、出生地等）
            birth_spacetime: 出生时空数据（经纬度、时区、精确时间等）
            device_entropy: 设备熵（可选）
            
        Returns:
            量子指纹的十六进制字符串
            
        Formula:
            H(bio_features || birth_spacetime || device_entropy)
        """
        # 规范化输入数据
        normalized_bio = json.dumps(bio_features, sort_keys=True, ensure_ascii=False)
        normalized_spacetime = json.dumps(birth_spacetime, sort_keys=True, ensure_ascii=False)
        
        # 设备熵（如果未提供则生成）
        if device_entropy is None:
            device_entropy = secrets.token_bytes(32)
        
        # 组合数据
        combined_data = (
            normalized_bio.encode('utf-8') +
            b'||' +
            normalized_spacetime.encode('utf-8') +
            b'||' +
            device_entropy
        )
        
        # 使用SHA3-256生成指纹
        hasher = sha3.sha3_256()
        hasher.update(combined_data)
        fingerprint = hasher.hexdigest()
        
        logger.info("生成量子指纹",
                   fingerprint_prefix=fingerprint[:8],
                   bio_features_keys=list(bio_features.keys()),
                   entropy_size=len(device_entropy))
        
        return fingerprint


class CryptoSigner:
    """Ed25519签名器"""
    
    def __init__(self, private_key: Optional[ed25519.Ed25519PrivateKey] = None):
        """
        初始化签名器
        
        Args:
            private_key: Ed25519私钥（如果不提供则生成新的）
        """
        if private_key is None:
            self.private_key = ed25519.Ed25519PrivateKey.generate()
            logger.warning("生成了新的Ed25519密钥对，请妥善保存私钥")
        else:
            self.private_key = private_key
        
        self.public_key = self.private_key.public_key()
    
    @classmethod
    def from_private_key_bytes(cls, private_key_bytes: bytes) -> 'CryptoSigner':
        """从私钥字节创建签名器"""
        private_key = ed25519.Ed25519PrivateKey.from_private_bytes(private_key_bytes)
        return cls(private_key)
    
    @classmethod
    def from_private_key_file(cls, key_path: Path, password: Optional[bytes] = None) -> 'CryptoSigner':
        """从私钥文件创建签名器"""
        with open(key_path, 'rb') as f:
            private_key = serialization.load_pem_private_key(
                f.read(),
                password=password,
                backend=default_backend()
            )
        if not isinstance(private_key, ed25519.Ed25519PrivateKey):
            raise ValueError("密钥类型必须是Ed25519")
        return cls(private_key)
    
    def save_private_key(self, key_path: Path, password: Optional[bytes] = None):
        """保存私钥到文件"""
        encryption_algorithm = (
            serialization.BestAvailableEncryption(password)
            if password
            else serialization.NoEncryption()
        )
        
        private_pem = self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=encryption_algorithm
        )
        
        key_path.parent.mkdir(parents=True, exist_ok=True)
        with open(key_path, 'wb') as f:
            f.write(private_pem)
        
        logger.info("私钥已保存", path=str(key_path), encrypted=password is not None)
    
    def get_public_key_hex(self) -> str:
        """获取公钥的十六进制表示"""
        public_bytes = self.public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )
        return public_bytes.hex()
    
    def sign(self, data: Union[str, bytes, Dict[str, Any]]) -> str:
        """
        对数据进行签名
        
        Args:
            data: 要签名的数据
            
        Returns:
            签名的十六进制字符串
        """
        # 规范化数据
        if isinstance(data, dict):
            data = json.dumps(data, sort_keys=True, ensure_ascii=False)
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        # 签名
        signature = self.private_key.sign(data)
        signature_hex = signature.hex()
        
        logger.debug("数据已签名",
                    signature_prefix=signature_hex[:16],
                    data_size=len(data))
        
        return signature_hex
    
    def verify(self, data: Union[str, bytes, Dict[str, Any]], 
              signature_hex: str) -> bool:
        """
        验证签名
        
        Args:
            data: 原始数据
            signature_hex: 签名的十六进制字符串
            
        Returns:
            签名是否有效
        """
        # 规范化数据
        if isinstance(data, dict):
            data = json.dumps(data, sort_keys=True, ensure_ascii=False)
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        # 转换签名
        signature = bytes.fromhex(signature_hex)
        
        try:
            self.public_key.verify(signature, data)
            return True
        except Exception:
            return False


class BlockchainAnchor:
    """区块链锚定器"""
    
    def __init__(self, rpc_url: Optional[str] = None, 
                private_key: Optional[str] = None):
        """
        初始化区块链锚定器
        
        Args:
            rpc_url: 区块链RPC URL
            private_key: 以太坊私钥（十六进制）
        """
        settings = get_settings()
        
        self.enabled = settings.blockchain_enabled
        if not self.enabled:
            logger.info("区块链锚定已禁用")
            return
        
        self.rpc_url = rpc_url or settings.blockchain_rpc_url
        self.private_key = private_key or settings.blockchain_private_key
        
        if not self.rpc_url:
            logger.warning("未配置区块链RPC URL，锚定功能不可用")
            self.enabled = False
            return
        
        # 初始化Web3
        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        
        if not self.w3.is_connected():
            logger.error("无法连接到区块链网络", rpc_url=self.rpc_url)
            self.enabled = False
            return
        
        # 设置账户
        if self.private_key:
            self.account = Account.from_key(self.private_key)
            self.address = self.account.address
            logger.info("区块链账户已配置", address=self.address)
        else:
            logger.warning("未配置私钥，只能读取区块链数据")
            self.account = None
            self.address = None
    
    def compute_hash(self, data: Dict[str, Any]) -> str:
        """
        计算数据的SHA3-256哈希
        
        Args:
            data: 要哈希的数据
            
        Returns:
            哈希值的十六进制字符串
        """
        # 规范化数据
        normalized = json.dumps(data, sort_keys=True, ensure_ascii=False)
        
        # 计算SHA3-256
        hasher = sha3.sha3_256()
        hasher.update(normalized.encode('utf-8'))
        
        return hasher.hexdigest()
    
    def anchor_to_chain(self, data_hash: str, 
                       metadata: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        将数据哈希锚定到区块链
        
        Args:
            data_hash: 数据哈希
            metadata: 元数据（可选）
            
        Returns:
            交易信息字典，包含tx_hash、block_number等
        """
        if not self.enabled or not self.account:
            logger.warning("区块链锚定不可用")
            return None
        
        try:
            # 构造数据
            anchor_data = {
                'hash': data_hash,
                'timestamp': datetime.utcnow().isoformat(),
                'metadata': metadata or {}
            }
            
            # 编码数据
            message = json.dumps(anchor_data, sort_keys=True)
            message_hash = encode_defunct(text=message)
            
            # 签名
            signed = self.account.sign_message(message_hash)
            
            # 获取nonce
            nonce = self.w3.eth.get_transaction_count(self.address)
            
            # 构造交易（这里简化为发送0 ETH到自己，数据放在input中）
            transaction = {
                'nonce': nonce,
                'to': self.address,
                'value': 0,
                'gas': 100000,
                'gasPrice': self.w3.eth.gas_price,
                'data': self.w3.to_hex(text=message),
                'chainId': self.w3.eth.chain_id
            }
            
            # 签名交易
            signed_txn = self.account.sign_transaction(transaction)
            
            # 发送交易
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            tx_hash_hex = tx_hash.hex()
            
            # 等待交易确认
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            result = {
                'tx_hash': tx_hash_hex,
                'block_number': receipt['blockNumber'],
                'block_hash': receipt['blockHash'].hex(),
                'gas_used': receipt['gasUsed'],
                'timestamp': datetime.utcnow().isoformat(),
                'network': settings.blockchain_network,
                'data_hash': data_hash
            }
            
            logger.info("数据已锚定到区块链",
                       tx_hash=tx_hash_hex[:16],
                       block=receipt['blockNumber'])
            
            return result
            
        except Exception as e:
            logger.error("区块链锚定失败", error=str(e))
            return None
    
    def verify_anchor(self, tx_hash: str) -> Optional[Dict[str, Any]]:
        """
        验证链上锚定
        
        Args:
            tx_hash: 交易哈希
            
        Returns:
            锚定信息
        """
        if not self.enabled:
            return None
        
        try:
            # 获取交易
            tx = self.w3.eth.get_transaction(tx_hash)
            receipt = self.w3.eth.get_transaction_receipt(tx_hash)
            
            # 解码数据
            if tx['input']:
                try:
                    message = self.w3.to_text(hexstr=tx['input'])
                    anchor_data = json.loads(message)
                except:
                    anchor_data = {'raw_input': tx['input']}
            else:
                anchor_data = {}
            
            return {
                'tx_hash': tx_hash,
                'block_number': receipt['blockNumber'],
                'block_hash': receipt['blockHash'].hex(),
                'from': tx['from'],
                'to': tx['to'],
                'timestamp': datetime.utcnow().isoformat(),
                'data': anchor_data,
                'verified': True
            }
            
        except Exception as e:
            logger.error("验证链上锚定失败", error=str(e), tx_hash=tx_hash)
            return None


# 便捷函数
def generate_quantum_fingerprint(bio_features: Dict[str, Any],
                               birth_spacetime: Dict[str, Any]) -> str:
    """生成量子指纹的便捷函数"""
    return QuantumFingerprint.generate(bio_features, birth_spacetime)


def sign_prediction(prediction_data: Dict[str, Any],
                   private_key_path: Optional[Path] = None) -> Tuple[str, str]:
    """
    签名预测数据
    
    Returns:
        (signature_hex, public_key_hex)
    """
    settings = get_settings()
    
    # 加载或生成签名器
    if private_key_path and private_key_path.exists():
        signer = CryptoSigner.from_private_key_file(private_key_path)
    else:
        signer = CryptoSigner()
        # 保存密钥
        key_path = settings.base_dir / "keys" / "signing_key.pem"
        signer.save_private_key(key_path)
    
    # 签名
    signature = signer.sign(prediction_data)
    public_key = signer.get_public_key_hex()
    
    return signature, public_key


def anchor_to_blockchain(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """将数据锚定到区块链的便捷函数"""
    anchor = BlockchainAnchor()
    data_hash = anchor.compute_hash(data)
    return anchor.anchor_to_chain(data_hash)


def verify_signature(data: Dict[str, Any], signature: str, 
                    public_key_hex: str) -> bool:
    """验证签名的便捷函数"""
    public_key_bytes = bytes.fromhex(public_key_hex)
    public_key = ed25519.Ed25519PublicKey.from_public_bytes(public_key_bytes)
    signer = CryptoSigner(ed25519.Ed25519PrivateKey.generate())  # 临时私钥
    signer.public_key = public_key
    return signer.verify(data, signature)


if __name__ == "__main__":
    # 测试量子指纹
    bio = {"name": "张三", "birth_date": "1990-01-01"}
    spacetime = {"longitude": 116.4074, "latitude": 39.9042, "timezone": "Asia/Shanghai"}
    fingerprint = generate_quantum_fingerprint(bio, spacetime)
    print(f"量子指纹: {fingerprint}")
    
    # 测试签名
    data = {"prediction": "未来一周运势良好", "confidence": 0.85}
    signature, public_key = sign_prediction(data)
    print(f"\n签名: {signature[:32]}...")
    print(f"公钥: {public_key[:32]}...")
    
    # 验证签名
    is_valid = verify_signature(data, signature, public_key)
    print(f"签名验证: {'有效' if is_valid else '无效'}")
    
    # 测试哈希
    anchor = BlockchainAnchor()
    data_hash = anchor.compute_hash(data)
    print(f"\nSHA3-256哈希: {data_hash}")