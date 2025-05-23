# F5-TTS API 接口文档

本文档描述了F5-TTS API服务的接口规范。

## 1. 服务概述

F5-TTS API服务提供文本到语音（TTS）推理功能，支持多种模型组、多语言和多音色合成。

服务基础URL: `http://127.0.0.1:8000`

## 2. 接口列表

### 2.1 获取可用模型列表

-   **接口路径**: `/tts/models`
-   **方法**: GET
-   **功能**: 获取API服务当前加载的可用模型组列表及其基本信息。
-   **请求参数**: 无
-   **响应**:
    -   状态码: `200 OK`
    -   格式: JSON 数组
    -   示例:
        ```json
        [
          {
            "name": "chinese_base",
            "language": ["zh"],
            "description": "支持中文的基础模型组",
            "models": ["F5TTS_Base_Chinese.pt"]
          },
          {
            "name": "japanese_manbo",
            "language": ["ja"],
            "description": "支持日语的 Manbo 模型组",
            "models": ["japanese_manbo.safetensors"]
          }
        ]
        ```
    -   字段说明:
        -   `name` (string): 模型组的名称，用于在推理请求中指定使用的模型组。
        -   `language` (array of string): 模型组支持的语言列表。
        -   `description` (string): 模型组的简要描述。
        -   `models` (array of string): 模型组中包含的具体模型文件列表（检查点文件名）。

### 2.2 执行TTS推理

-   **接口路径**: `/tts/synthesize`
-   **方法**: POST
-   **功能**: 根据提供的文本和参考音频生成语音。支持单声音和多音色合成。
-   **请求参数**:
    -   格式: JSON 对象
    -   字段说明:
        -   `group_name` (string, 必需): 指定使用的模型组名称，必须是 `/tts/models` 接口返回列表中的一个模型组名称。
        -   `model_name` (string, 必需): 指定使用的模型组中具体模型的名称（检查点文件名），必须是指定模型组 `models` 列表中的一个。
        -   `ref_audio` (string, 必需): 参考音频数据。可以是文件路径（API服务可访问的本地路径）或Base64编码的音频数据（推荐）。Base64编码数据应包含data URL前缀，例如 `data:audio/wav;base64,...`。
        -   `ref_text` (string, 可选): 参考音频对应的文本。如果提供，将用于辅助推理；如果为空字符串，API服务可能会尝试自动转录参考音频（取决于底层实现）。默认为空字符串。
        -   `gen_text` (string, 必需): 需要生成语音的文本。对于多音色合成，文本中可以使用 `[voice_name]` 标记来切换音色。
        -   `ref_language` (string, 必需): 参考音频的语言。必须是指定模型组支持的语言之一。
        -   `gen_language` (string, 必需): 生成文本的语言。必须是指定模型组支持的语言之一。
        -   `voices` (object, 可选): 多声音配置。一个字典，键是声音名称（用于在 `gen_text` 中标记），值是包含该声音参考音频和文本的字典。默认为空字典。
            -   `voice_name` (string): 声音名称。
                -   `ref_audio` (string, 必需): 该声音的参考音频数据（文件路径或Base64编码）。
                -   `ref_text` (string, 可选): 该声音参考音频对应的文本。默认为空字符串。
                -   `ref_language` (string, 必需): 该声音参考音频的语言。
            -   `voice_name` (string): 声音名称。
                -   `ref_audio` (string, 必需): 该声音的参考音频数据（文件路径或Base64编码）。
                -   `ref_text` (string, 可选): 该声音参考音频对应的文本。默认为空字符串。

-   **响应**:
    -   状态码: `200 OK`
    -   格式: 音频数据 (例如，`audio/wav`)
    -   响应体: 生成的音频文件的二进制数据。

    -   **请求示例 (单声音)**:
    ```json
    {
      "group_name": "chinese_base",
      "model_name": "F5TTS_Base_Chinese.pt",
      "ref_audio": "data:audio/wav;base64,..." , // Base64编码的参考音频数据
      "ref_text": "这是参考音频的文本。",
      "ref_language": "ja", // 参考音频的语言
      "gen_text": "这是要生成的文本。",
      "gen_language": "zh" // 生成文本的语言
    }
    ```

    -   **请求示例 (多音色)**:
    ```json
    {
      "group_name": "chinese_base",
      "model_name": "F5TTS_Base_Chinese.pt",
      "ref_audio": "data:audio/wav;base64,...", // 默认声音 (main) 的Base64参考音频
      "ref_text": "", // 默认声音的参考文本
      "ref_language": "ja", // 默认声音参考音频的语言
      "gen_text": "这是默认声音。[voice1]这是第一个声音。[main]又回到默认声音。",
      "gen_language": "zh", // 生成文本的语言
      "voices": {
        "voice1": {
          "ref_audio": "data:audio/wav;base64,...", // 声音 'voice1' 的Base64参考音频
          "ref_text": "",
          "ref_language": "en" // 该声音参考音频的语言
        },
        "voice2": {
          "ref_audio": "/path/to/voice2.wav", // 声音 'voice2' 的文件路径参考音频
          "ref_text": "这是第二个声音的参考文本。",
          "ref_language": "zh" // 该声音参考音频的语言
        }
      }
    }
    ```
    **注意**: 在多音色合成中，`ref_audio` 和 `ref_text` 字段用于定义默认声音（`[main]` 标记或未标记的文本块）。`voices` 字典用于定义其他声音。参考音频可以通过Base64编码或文件路径提供。

## 3. 错误响应

API接口在发生错误时会返回相应的HTTP状态码和JSON格式的错误详情。

-   **`400 Bad Request`**: 请求参数无效或处理请求时发生错误（例如，参考音频文件不存在，模型组不支持指定的语言，未知的声音标记等）。
    -   响应体示例: `{"detail": "错误信息描述"}`
-   **`422 Unprocessable Entity`**: 请求体格式正确，但包含无效的数据（例如，必需字段缺失，数据类型不匹配等）。这通常是Pydantic模型验证失败导致的。
    -   响应体示例: `{"detail": [{"loc": ["body", "field_name"], "msg": "错误信息", "type": "错误类型"}]}`
-   **`500 Internal Server Error`**: API服务内部发生未知错误。
    -   响应体示例: `{"detail": "推理失败: 错误信息描述"}`

## 4. 依赖

运行API服务需要安装以下Python库：

-   `fastapi`
-   `uvicorn`
-   `PyYAML`
-   `pydantic`
-   `soundfile`
-   `numpy`
-   `pykakasi` (用于日语处理)
-   `pydub` (用于音频处理)
-   `requests` (用于测试或客户端调用)
-   `torch`, `torchaudio` (F5-TTS模型依赖)
-   F5-TTS项目本身的依赖

请确保您的环境中安装了所有必需的依赖，特别是与您的硬件兼容的PyTorch版本。
