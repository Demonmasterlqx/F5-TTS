import logging
from fastapi import APIRouter, Request, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ValidationError
import io
import os # 导入 os 模块用于检查文件是否存在

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

class SynthesizeRequest(BaseModel):
    model_name: str
    ref_audio: str # 文件路径或Base64编码
    ref_text: str = "" # 可选，默认为空字符串
    gen_text: str
    language: str = None # 可选，指定推理语言
    voices: dict = {} # 可选，多声音配置

@router.get("/models")
async def get_models(request: Request):
    """
    获取可用模型列表
    """
    logger.info("Received request for /models")
    model_manager = request.app.state.tts_service.model_manager
    models_info = [
        {"name": config.get("name"), "language": config.get("language"), "description": config.get("description")}
        for config in model_manager.model_configs.values()
    ]
    logger.info(f"Returning {len(models_info)} models")
    return models_info

@router.post("/synthesize")
async def synthesize_audio(request: Request, synth_request: SynthesizeRequest):
    """
    执行TTS推理
    """
    # logger.info(f"Received synthesize request with parameters: model_name={synth_request.model_name}, ref_audio={'<present>' if synth_request.ref_audio else '<empty>'}, ref_text='{synth_request.ref_text}', gen_text='{synth_request.gen_text}', language='{synth_request.language}', voices={synth_request.voices}")
    logger.info("Received synthesize request")
    tts_service = request.app.state.tts_service

    try:
        logger.info("Calling tts_service.synthesize...")
        audio_data_bytes = tts_service.synthesize(
            synth_request.model_name,
            synth_request.ref_audio,
            synth_request.ref_text,
            synth_request.gen_text,
            synth_request.language, # 传递语言参数
            synth_request.voices # 传递 voices 参数
        )

        # 返回音频数据
        logger.info("Synthesize successful, returning audio data")
        return StreamingResponse(io.BytesIO(audio_data_bytes), media_type="audio/wav")

    except ValueError as e:
        # 捕获 tts_service 中抛出的 ValueError，通常是模型或语言不支持
        # logger.error(f"ValueError during synthesize: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        # 捕获其他未知错误
        # logger.error(f"Unexpected error during synthesize: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"推理失败: {e}")
