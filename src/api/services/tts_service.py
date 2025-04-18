import base64
import io
import soundfile as sf
import numpy as np
import tempfile
import os
import pykakasi # 导入 pykakasi
import re # 导入 re 模块用于解析文本标记

from src.api.services.model_manager import ModelManager
from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text, load_vocoder, device, mel_spec_type

def convert_kanji_to_kana(text):
    """
    将日语文本中的汉字转换为平假名
    
    参数:
        text: 包含汉字的日语文本
        
    返回:
        转换后的文本，汉字被替换为平假名
    """
    kks = pykakasi.kakasi()
    result = kks.convert(text)
    
    # 将转换结果组合成一个字符串
    converted_text = ''
    for item in result:
        # 使用平假名替换汉字
        converted_text += item['hira']
    
    return converted_text

class TtsService:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        # 假设 vocoder 是共享的，可以在服务启动时加载一次
        self.vocoder = load_vocoder(device=device) # 使用默认参数加载 vocoder

    def synthesize(self, model_name: str, ref_audio_data: str, ref_text: str, gen_text: str, language: str = None, voices: dict = None):
        """
        执行TTS推理

        Args:
            model_name: 使用的模型名称
            ref_audio_data: 参考音频数据 (Base64编码或文件路径)
            ref_text: 参考音频文本 (可选)
            gen_text: 要生成的文本
            language: 指定推理语言 (可选)
            voices: 多声音配置 (可选)

        Returns:
            生成的音频数据 (bytes)
        """
        model_instance = self.model_manager.get_model(model_name)
        if not model_instance:
            raise ValueError(f"模型 '{model_name}' 未找到或加载失败。")

        model_config = self.model_manager.get_model_config(model_name)
        if not model_config:
             raise ValueError(f"模型配置 '{model_name}' 未找到。")

        supported_languages = model_config.get("language", []) # 从模型配置获取支持的语言列表

        # 确定使用的语言
        if language:
            if language not in supported_languages:
                raise ValueError(f"模型 '{model_name}' 不支持语言 '{language}'。支持的语言: {', '.join(supported_languages)}")
            used_language = language
        elif supported_languages:
            used_language = supported_languages[0] # 使用模型配置中的第一个语言作为默认值
        else:
            raise ValueError(f"模型 '{model_name}' 未指定支持的语言。")

        # 处理参考音频数据 (默认声音)
        if ref_audio_data.startswith("data:"):
            # Base64编码的数据
            header, encoded = ref_audio_data.split(",", 1)
            audio_bytes = base64.b64decode(encoded)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio_file:
                tmp_audio_file.write(audio_bytes)
                default_ref_audio_path = tmp_audio_file.name
        else:
            # 文件路径
            default_ref_audio_path = ref_audio_data

        # 预处理默认声音的参考音频和文本
        default_processed_ref_audio_path, default_processed_ref_text = preprocess_ref_audio_text(default_ref_audio_path, ref_text)

        # 如果是日语，将汉字转换为片假名并打印处理前后的文本 (默认声音)
        if used_language == "ja":
            print(f"原始 default_ref_text: {ref_text}")
            default_processed_ref_text = convert_kanji_to_kana(default_processed_ref_text)
            print(f"转换后 default_ref_text: {default_processed_ref_text}")

        # 处理多声音配置
        processed_voices = {}
        if voices:
            for voice_name, voice_config in voices.items():
                if "ref_audio" not in voice_config:
                    raise ValueError(f"声音 '{voice_name}' 配置缺少 'ref_audio'。")
                
                voice_ref_audio_data = voice_config["ref_audio"]
                voice_ref_text = voice_config.get("ref_text", "")

                # 处理参考音频数据 (其他声音)
                if voice_ref_audio_data.startswith("data:"):
                    header, encoded = voice_ref_audio_data.split(",", 1)
                    audio_bytes = base64.b64decode(encoded)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio_file:
                        tmp_audio_file.write(audio_bytes)
                        voice_ref_audio_path = tmp_audio_file.name
                else:
                    voice_ref_audio_path = voice_ref_audio_data

                # 预处理其他声音的参考音频和文本
                processed_voice_ref_audio_path, processed_voice_ref_text = preprocess_ref_audio_text(voice_ref_audio_path, voice_ref_text)

                # 如果是日语，将汉字转换为片假名并打印处理前后的文本 (其他声音)
                if used_language == "ja":
                    print(f"原始 {voice_name}_ref_text: {voice_ref_text}")
                    processed_voice_ref_text = convert_kanji_to_kana(processed_voice_ref_text)
                    print(f"转换后 {voice_name}_ref_text: {processed_voice_ref_text}")

                processed_voices[voice_name] = {
                    "ref_audio_path": processed_voice_ref_audio_path,
                    "ref_text": processed_voice_ref_text
                }

        # 解析文本中的声音标记并分割文本
        generated_audio_segments = []
        reg1 = r"(?=\[\w+\])"
        chunks = re.split(reg1, gen_text)
        reg2 = r"\[(\w+)\]"

        temp_files_to_clean = [] # 记录需要清理的临时文件

        try:
            for text_chunk in chunks:
                if not text_chunk.strip():
                    continue

                match = re.match(reg2, text_chunk)
                if match:
                    voice_name = match[1]
                    current_text = re.sub(reg2, "", text_chunk).strip()
                else:
                    voice_name = "main" # 默认声音
                    current_text = text_chunk.strip()

                if not current_text:
                    continue

                # 获取当前声音的参考音频和文本
                if voice_name == "main":
                    current_ref_audio_path = default_processed_ref_audio_path
                    current_ref_text = default_processed_ref_text
                elif voice_name in processed_voices:
                    current_ref_audio_path = processed_voices[voice_name]["ref_audio_path"]
                    current_ref_text = processed_voices[voice_name]["ref_text"]
                else:
                    raise ValueError(f"未知的声音标记: [{voice_name}]")

                # 如果是日语，将汉字转换为片假名并打印处理前后的文本 (当前文本块)
                if used_language == "ja":
                    print(f"原始 text_chunk: {current_text}")
                    current_text = convert_kanji_to_kana(current_text)
                    print(f"转换后 text_chunk: {current_text}")

                # 执行推理
                audio_segment, final_sample_rate, _ = infer_process(
                    current_ref_audio_path,
                    current_ref_text,
                    current_text,
                    model_instance,
                    self.vocoder,
                    mel_spec_type=mel_spec_type,
                    language=used_language
                )
                generated_audio_segments.append(audio_segment)

            # 将所有生成的音频片段拼接起来
            if generated_audio_segments:
                final_wave = np.concatenate(generated_audio_segments)
                audio_buffer = io.BytesIO()
                sf.write(audio_buffer, final_wave, final_sample_rate, format='wav')
                audio_data_bytes = audio_buffer.getvalue()
                return audio_data_bytes
            else:
                return b"" # 没有生成音频片段

        finally:
            # 清理临时文件 (默认声音)
            if ref_audio_data.startswith("data:") and os.path.exists(default_ref_audio_path):
                os.remove(default_ref_audio_path)
            # 清理临时文件 (其他声音)
            if voices:
                for voice_config in processed_voices.values():
                    if voice_config["ref_audio_path"].startswith(tempfile.gettempdir()): # 只清理临时文件
                         if os.path.exists(voice_config["ref_audio_path"]):
                            os.remove(voice_config["ref_audio_path"])


# 示例用法 (需要实例化 ModelManager 和 TtsService)
if __name__ == '__main__':
    # 假设已经创建了 config/models 目录和模型配置文件
    model_manager = ModelManager()
    tts_service = TtsService(model_manager)

    # 示例推理请求 (单声音)
    try:
        sample_ref_audio = "src/f5_tts/infer/examples/basic/basic_ref_en.wav"
        gen_text_sample = "这是一个测试文本。"
        model_name_sample = "F5TTS_Base_Chinese"

        print(f"正在使用模型 '{model_name_sample}' 进行单声音推理...")
        audio_output = tts_service.synthesize(model_name_sample, sample_ref_audio, "", gen_text_sample, language="zh")

        output_filename = "output/test_single_voice_output.wav"
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        with open(output_filename, "wb") as f:
            f.write(audio_output)
        print(f"生成的单声音音频已保存到: {output_filename}")

    except Exception as e:
        print(f"单声音推理失败: {e}")

    # 示例推理请求 (多声音)
    try:
        sample_ref_audio_main = "src/f5_tts/infer/examples/multi/main.flac" # 默认声音参考音频
        sample_ref_text_main = "" # 默认声音参考文本

        voices_config = {
            "town": {
                "ref_audio": "src/f5_tts/infer/examples/multi/town.flac",
                "ref_text": ""
            },
            "country": {
                "ref_audio": "src/f5_tts/infer/examples/multi/country.flac",
                "ref_text": ""
            }
        }
        gen_text_multi = "A Town Mouse and a Country Mouse were acquaintances, and the Country Mouse one day invited his friend to come and see him at his home in the fields. The Town Mouse came, and they sat down to a dinner of barleycorns and roots, the latter of which had a distinctly earthy flavour. The fare was not much to the taste of the guest, and presently he broke out with [town] “My poor dear friend, you live here no better than the ants. Now, you should just see how I fare! My larder is a regular horn of plenty. You must come and stay with me, and I promise you you shall live on the fat of the land.” [main] So when he returned to town he took the Country Mouse with him, and showed him into a larder containing flour and oatmeal and figs and honey and dates. The Country Mouse had never seen anything like it, and sat down to enjoy the luxuries his friend provided: but before they had well begun, the door of the larder opened and someone came in. The two Mice scampered off and hid themselves in a narrow and exceedingly uncomfortable hole. Presently, when all was quiet, they ventured out again; but someone else came in, and off they scuttled again. This was too much for the visitor. [country] “Goodbye,” [main] said he, [country] “I’m off. You live in the lap of luxury, I can see, but you are surrounded by dangers; whereas at home I can enjoy my simple dinner of roots and corn in peace.”"
        model_name_sample = "F5TTS_Base_Chinese" # 替换为你支持多音色的模型名称

        print(f"\n正在使用模型 '{model_name_sample}' 进行多声音推理...")
        audio_output_multi = tts_service.synthesize(model_name_sample, sample_ref_audio_main, sample_ref_text_main, gen_text_multi, language="zh", voices=voices_config)

        output_filename_multi = "output/test_multi_voice_output.wav"
        os.makedirs(os.path.dirname(output_filename_multi), exist_ok=True)
        with open(output_filename_multi, "wb") as f:
            f.write(audio_output_multi)
        print(f"生成的多声音音频已保存到: {output_filename_multi}")

    except Exception as e:
        print(f"多声音推理失败: {e}")
