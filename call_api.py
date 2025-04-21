import httpx
import asyncio
import os
import base64

# 定义 API 的基础 URL
API_BASE_URL = "http://localhost:8000"

async def call_synthesize_api(text: str, model_name: str = "chinese_base", ref_audio_path: str = None, output_filename: str = "generated_audio.wav"):
    """
    调用 /tts/synthesize API 并保存生成的音频
    """

    with open("/home/lqx/code/F5-TTS/data/ja_manbo/006hqbwh6x5rephh4dx052rkk9tv80yxn9.wav", "rb") as f:
        audio_data = f.read()

    base64code=base64.b64encode(audio_data).decode("utf-8")
    base64code=f'data:audio/wav;base64,{base64code}'

    print(f"base64code: {base64code[:200]}")

    api_url = f"{API_BASE_URL}/tts/synthesize"
    test_payload = {
        "group_name": "japanese_multiple_group", # 根据实际配置修改
        "model_name": "model_21999905.pt",
        "ref_audio": base64code, # 如果需要克隆声音，提供参考音频路径或 Base64 编码
        "ref_text": "", # 如果提供了 ref_audio，这里可以填写参考音频对应的文本
        "gen_text": "私はなにもかもが普通ですけど、トレーナーと出会えた強運だけは、特別だと思うんですよね！うひひ。",
        "language": "ja", # 根据实际配置修改
        "voices": {} # 多声音配置，如果需要
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(api_url, json=test_payload)

        if response.status_code == 200:
            # 将返回的音频数据保存到文件
            with open(output_filename, "wb") as f:
                f.write(response.content)
            print(f"音频生成成功，已保存到 {output_filename}")
        else:
            print(f"API 请求失败，状态码: {1}")
            print(f"错误信息: {1}")

    except httpx.RequestError as e:
        print(f"API 请求发生错误: {e}")
    except Exception as e:
        print(f"发生未知错误: {e}")

async def get_model():
    api_url = f"{API_BASE_URL}/tts/models"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(api_url)

        if response.status_code == 200:
            models_info = response.json()
            print("可用模型列表:")
            for model in models_info:
                print(model)
        else:
            print(f"API 请求失败，状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
    except httpx.RequestError as e:
        print(f"API 请求发生错误: {e}")
    except Exception as e:
        print(f"发生未知错误: {e}")


# 示例用法
if __name__ == "__main__":
    # 需要合成的文本
    text_to_synthesize = "你好，这是一个简单的 API 调用测试。"

    # 示例：获取可用模型列表
    asyncio.run(get_model())

    # 调用 API 并保存音频
    asyncio.run(call_synthesize_api(text_to_synthesize))

    # 示例：调用 API 并指定模型和输出文件名
    # asyncio.run(call_synthesize_api("测试使用不同的模型。", model_name="japanese_model", output_filename="japanese_test.wav"))

    # 示例：调用 API 并进行声音克隆 (需要提供参考音频路径)
    # asyncio.run(call_synthesize_api("测试声音克隆。", ref_audio_path="path/to/your/ref_audio.wav", output_filename="cloned_voice_test.wav"))

# 如何运行此脚本：
# 1. 确保 FastAPI 应用正在运行 (例如，使用 `uvicorn src.api.main:app --reload`)
# 2. 在终端中导航到项目根目录
# 3. 运行脚本: `python call_api.py`
# 4. 生成的音频文件将保存在项目根目录下
