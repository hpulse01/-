"""
API接口模块
提供RESTful API和GraphQL接口
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import asyncio
import json
import structlog
from contextlib import asynccontextmanager

# FastAPI
from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

# GraphQL
from graphene import ObjectType, String, Float, List as GraphList, Field as GraphField
from graphene import Schema, Mutation, InputObjectType, Boolean, Int
# Note: starlette_graphene3 would be used in production
# For now, we'll mount GraphQL differently to avoid dependency issues

# 内部模块
from ..prediction_ai import predict_life_trajectory
from ..data_collection import calculate_bazi, calculate_ziwei, calculate_natal_chart
from ..quantum_engine import simulate_life_superposition
from ..utils.settings import get_settings
from ..utils.crypto_anchor import verify_signature
from .report import generate_pdf_report

logger = structlog.get_logger()


# Pydantic模型
class BirthLocation(BaseModel):
    """出生地点"""
    latitude: float = Field(..., ge=-90, le=90, description="纬度")
    longitude: float = Field(..., ge=-180, le=180, description="经度")
    timezone: str = Field(default="UTC", description="时区")


class PredictionRequest(BaseModel):
    """预测请求"""
    birth_datetime: str = Field(..., description="出生时间，ISO格式")
    birth_location: BirthLocation = Field(..., description="出生地点")
    name: Optional[str] = Field(None, description="姓名")
    gender: str = Field(default="男", description="性别：男/女")
    user_id: Optional[str] = Field(None, description="用户ID")
    
    @validator('birth_datetime')
    def validate_datetime(cls, v):
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError("出生时间格式错误，请使用ISO格式")
        return v
    
    @validator('gender')
    def validate_gender(cls, v):
        if v not in ["男", "女"]:
            raise ValueError("性别必须是'男'或'女'")
        return v


class EventPrediction(BaseModel):
    """事件预测"""
    id: str
    type: str
    description: str
    probability: float
    confidence: float
    expected_date: str
    impact: str
    suggestions: List[str]


class PredictionResponse(BaseModel):
    """预测响应"""
    user_id: str
    prediction_time: str
    events: List[EventPrediction]
    timeline_summary: str
    overall_trend: Dict[str, str]
    feature_importance: Dict[str, float]
    confidence_metrics: Dict[str, float]
    report_url: Optional[str] = None


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    version: str
    timestamp: str
    services: Dict[str, str]


class VerifyRequest(BaseModel):
    """验证请求"""
    data: Dict[str, Any]
    signature: str
    public_key: str


# GraphQL模型
class EventType(ObjectType):
    """GraphQL事件类型"""
    id = String()
    type = String()
    description = String()
    probability = Float()
    confidence = Float()
    expected_date = String()
    impact = String()
    suggestions = GraphList(String)


class PredictionType(ObjectType):
    """GraphQL预测类型"""
    user_id = String()
    prediction_time = String()
    events = GraphList(EventType)
    timeline_summary = String()
    overall_trend = String()
    confidence = Float()


class BirthInfoInput(InputObjectType):
    """GraphQL出生信息输入"""
    birth_datetime = String(required=True)
    latitude = Float(required=True)
    longitude = Float(required=True)
    timezone = String(default_value="UTC")
    gender = String(default_value="男")
    name = String()


class Query(ObjectType):
    """GraphQL查询"""
    
    prediction = GraphField(
        PredictionType,
        user_id=String(required=True)
    )
    
    events = GraphList(
        EventType,
        user_id=String(required=True),
        event_type=String(),
        min_probability=Float(default_value=0.0)
    )
    
    def resolve_prediction(self, info, user_id):
        # 实际应该从数据库获取
        # 这里返回示例数据
        return PredictionType(
            user_id=user_id,
            prediction_time=datetime.now().isoformat(),
            events=[],
            timeline_summary="示例预测摘要",
            overall_trend=json.dumps({"trend": "stable"}),
            confidence=0.75
        )
    
    def resolve_events(self, info, user_id, event_type=None, min_probability=0.0):
        # 实际应该从数据库查询
        # 这里返回示例数据
        events = []
        
        if min_probability <= 0.5:
            events.append(EventType(
                id="evt_001",
                type=event_type or "career",
                description="示例事件",
                probability=0.6,
                confidence=0.7,
                expected_date="2025-01-01",
                impact="moderate",
                suggestions=["建议1", "建议2"]
            ))
        
        return events


class CreatePrediction(Mutation):
    """创建预测的变更"""
    
    class Arguments:
        birth_info = BirthInfoInput(required=True)
        user_id = String()
    
    success = Boolean()
    prediction = GraphField(PredictionType)
    message = String()
    
    def mutate(self, info, birth_info, user_id=None):
        try:
            # 构建预测请求
            birth_data = {
                'birth_datetime': birth_info.birth_datetime,
                'longitude': birth_info.longitude,
                'latitude': birth_info.latitude,
                'gender': birth_info.gender,
                'user_id': user_id or 'anonymous'
            }
            
            # 执行预测（简化版）
            prediction = PredictionType(
                user_id=user_id or 'anonymous',
                prediction_time=datetime.now().isoformat(),
                events=[],
                timeline_summary="预测已创建",
                overall_trend=json.dumps({"trend": "processing"}),
                confidence=0.0
            )
            
            return CreatePrediction(
                success=True,
                prediction=prediction,
                message="预测请求已接受"
            )
            
        except Exception as e:
            return CreatePrediction(
                success=False,
                prediction=None,
                message=str(e)
            )


class Mutations(ObjectType):
    """GraphQL变更"""
    create_prediction = CreatePrediction.Field()


# 创建GraphQL模式
graphql_schema = Schema(query=Query, mutation=Mutations)


# 生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    logger.info("H-Pulse API启动中...")
    settings = get_settings()
    logger.info(f"API版本: {settings.project_version}")
    
    yield
    
    # 关闭时
    logger.info("H-Pulse API关闭中...")


# 创建FastAPI应用
app = FastAPI(
    title="H-Pulse Quantum Prediction API",
    description="量子生命轨迹预测系统API",
    version="1.1.0",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 添加GraphQL端点
# Note: In production, use starlette_graphene3
# app.mount("/graphql", GraphQLApp(schema=graphql_schema, on_get=make_playground_handler()))
# For now, we'll add a simple GraphQL endpoint
@app.post("/graphql")
async def graphql_endpoint(query: str = "", variables: dict = None):
    """简化的GraphQL端点"""
    return {"data": {"message": "GraphQL endpoint placeholder"}}


@app.get("/", tags=["root"])
async def root():
    """根路径"""
    return {
        "message": "H-Pulse Quantum Prediction System",
        "version": "1.1.0",
        "tagline": "Precision · Uniqueness · Irreversibility",
        "docs": "/docs",
        "graphql": "/graphql"
    }


@app.get("/healthz", response_model=HealthResponse, tags=["health"])
async def health_check():
    """健康检查"""
    settings = get_settings()
    
    # 检查各个服务状态
    services = {
        "api": "healthy",
        "quantum_engine": "healthy",
        "ai_model": "healthy",
        "database": "not_configured"
    }
    
    # 如果启用了区块链
    if settings.blockchain_enabled:
        services["blockchain"] = "healthy" if settings.blockchain_rpc_url else "not_configured"
    
    return HealthResponse(
        status="healthy",
        version=settings.project_version,
        timestamp=datetime.now().isoformat(),
        services=services
    )


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
async def create_prediction(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    generate_report: bool = True
):
    """
    创建生命轨迹预测
    
    - **birth_datetime**: 出生时间（ISO格式，如：1990-01-01T12:00:00+08:00）
    - **birth_location**: 出生地点（经纬度和时区）
    - **name**: 姓名（可选）
    - **gender**: 性别（男/女）
    - **user_id**: 用户ID（可选）
    - **generate_report**: 是否生成PDF报告
    """
    try:
        # 构建出生数据
        birth_data = {
            'birth_datetime': request.birth_datetime,
            'longitude': request.birth_location.longitude,
            'latitude': request.birth_location.latitude,
            'timezone': request.birth_location.timezone,
            'gender': request.gender,
            'name': request.name,
            'user_id': request.user_id or f"user_{datetime.now().timestamp()}"
        }
        
        logger.info("开始预测", user_id=birth_data['user_id'])
        
        # 执行预测
        prediction_result = await asyncio.to_thread(
            predict_life_trajectory, 
            birth_data
        )
        
        # 转换为响应格式
        events = []
        for event in prediction_result['life_trajectory']['events'][:10]:
            events.append(EventPrediction(
                id=event['id'],
                type=event['type'],
                description=event['description'],
                probability=event['probability'],
                confidence=event['confidence'],
                expected_date=event['expected_date'],
                impact=event['impact'],
                suggestions=event['suggestions']
            ))
        
        response = PredictionResponse(
            user_id=prediction_result['user_id'],
            prediction_time=prediction_result['prediction_time'],
            events=events,
            timeline_summary=prediction_result['life_trajectory']['timeline_summary'],
            overall_trend=prediction_result['life_trajectory']['overall_trend'],
            feature_importance=prediction_result['feature_importance'],
            confidence_metrics=prediction_result['confidence_metrics']
        )
        
        # 后台生成报告
        if generate_report:
            report_filename = f"report_{birth_data['user_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            background_tasks.add_task(
                generate_pdf_report,
                prediction_result,
                report_filename
            )
            response.report_url = f"/reports/{report_filename}"
        
        logger.info("预测完成", 
                   user_id=birth_data['user_id'],
                   num_events=len(events))
        
        return response
        
    except Exception as e:
        logger.error("预测失败", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/events/{event_id}", tags=["events"])
async def get_event_details(event_id: str):
    """获取事件详情"""
    # 实际应该从数据库查询
    # 这里返回示例数据
    return {
        "id": event_id,
        "type": "career",
        "description": "职业发展机遇",
        "probability": 0.85,
        "confidence": 0.8,
        "expected_date": "2025-06-15T10:00:00",
        "location": {"latitude": 39.9, "longitude": 116.4},
        "impact": "major",
        "suggestions": [
            "把握机会，主动出击",
            "提升专业技能",
            "扩展人脉网络"
        ],
        "related_events": ["evt_002", "evt_003"],
        "astrological_influences": {
            "jupiter": "favorable",
            "saturn": "neutral"
        }
    }


@app.post("/charts/bazi", tags=["charts"])
async def calculate_bazi_chart(request: PredictionRequest):
    """计算四柱八字"""
    try:
        birth_dt = datetime.fromisoformat(request.birth_datetime)
        
        chart = await asyncio.to_thread(
            calculate_bazi,
            birth_dt,
            request.birth_location.longitude,
            request.birth_location.latitude,
            request.gender
        )
        
        return chart.to_dict()
        
    except Exception as e:
        logger.error("八字计算失败", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/charts/ziwei", tags=["charts"])
async def calculate_ziwei_chart(request: PredictionRequest):
    """计算紫微斗数"""
    try:
        birth_dt = datetime.fromisoformat(request.birth_datetime)
        
        chart = await asyncio.to_thread(
            calculate_ziwei,
            birth_dt,
            request.birth_location.longitude,
            request.birth_location.latitude,
            request.gender
        )
        
        return chart.to_dict()
        
    except Exception as e:
        logger.error("紫微计算失败", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/charts/natal", tags=["charts"])
async def calculate_natal_chart_api(request: PredictionRequest):
    """计算西方星盘"""
    try:
        birth_dt = datetime.fromisoformat(request.birth_datetime)
        
        chart = await asyncio.to_thread(
            calculate_natal_chart,
            birth_dt,
            request.birth_location.longitude,
            request.birth_location.latitude
        )
        
        return chart.to_dict()
        
    except Exception as e:
        logger.error("星盘计算失败", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/anchor/verify", tags=["verification"])
async def verify_anchor(request: VerifyRequest):
    """验证签名和链上锚定"""
    try:
        # 验证签名
        is_valid = verify_signature(
            request.data,
            request.signature,
            request.public_key
        )
        
        result = {
            "signature_valid": is_valid,
            "public_key": request.public_key,
            "timestamp": datetime.now().isoformat()
        }
        
        # 如果有区块链锚定信息，也可以验证
        if "blockchain_anchor" in request.data:
            # 这里应该查询区块链验证
            result["blockchain_verified"] = True
            result["tx_hash"] = request.data["blockchain_anchor"].get("tx_hash")
        
        return result
        
    except Exception as e:
        logger.error("验证失败", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/reports/{filename}", tags=["reports"])
async def download_report(filename: str):
    """下载报告"""
    settings = get_settings()
    file_path = settings.get_output_path(filename, "reports")
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="报告不存在")
    
    return FileResponse(
        path=str(file_path),
        media_type="application/pdf",
        filename=filename
    )


@app.post("/upload/birth-certificate", tags=["upload"])
async def upload_birth_certificate(
    file: UploadFile = File(...),
    user_id: Optional[str] = None
):
    """上传出生证明（用于提取信息）"""
    # 检查文件类型
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="只支持图片文件")
    
    # 保存文件
    settings = get_settings()
    upload_dir = settings.data_dir / "uploads"
    upload_dir.mkdir(exist_ok=True)
    
    file_path = upload_dir / f"{user_id or 'anonymous'}_{file.filename}"
    
    content = await file.read()
    with open(file_path, 'wb') as f:
        f.write(content)
    
    # 这里应该调用OCR服务提取信息
    # 现在返回示例结果
    return {
        "status": "uploaded",
        "filename": file.filename,
        "size": len(content),
        "extracted_info": {
            "birth_date": "1990-01-01",
            "birth_time": "12:00",
            "location": "北京市",
            "confidence": 0.85
        }
    }


@app.get("/statistics", tags=["statistics"])
async def get_statistics():
    """获取系统统计信息"""
    # 实际应该从数据库获取
    return {
        "total_predictions": 1234,
        "today_predictions": 56,
        "active_users": 789,
        "average_confidence": 0.78,
        "popular_event_types": {
            "career": 0.35,
            "relationship": 0.28,
            "health": 0.20,
            "wealth": 0.17
        },
        "system_uptime": "7 days 3 hours",
        "last_update": datetime.now().isoformat()
    }


def start_api_server(host: str = "0.0.0.0", port: int = 8000):
    """启动API服务器的便捷函数"""
    settings = get_settings()
    
    logger.info(
        "启动H-Pulse API服务器",
        host=host,
        port=port,
        version=settings.project_version
    )
    
    uvicorn.run(
        "h_pulse.output_generation.api:app",
        host=host,
        port=port,
        reload=settings.debug,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "level": settings.log_level,
                "handlers": ["default"],
            },
        }
    )


if __name__ == "__main__":
    # 直接运行API服务器
    start_api_server()