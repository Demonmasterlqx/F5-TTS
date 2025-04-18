from fastapi import FastAPI
from src.api.services.model_manager import ModelManager
from src.api.services.tts_service import TtsService
from src.api.routes import tts

app = FastAPI()

# 初始化 ModelManager 和 TtsService
model_manager = ModelManager()
tts_service = TtsService(model_manager)

# 将 tts_service 实例添加到 app.state，以便在路由中访问
app.state.tts_service = tts_service

# 包含 TTS 路由
app.include_router(tts.router, prefix="/tts")

@app.get("/")
def read_root():
    return {"message": "F5-TTS API Service"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
