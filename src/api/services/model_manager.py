import torch
import importlib
import sys
import os # 导入 os 模块

from src.api.utils.config_parser import load_model_configs, load_config
from f5_tts.model import CFM
from f5_tts.infer.utils_infer import load_checkpoint, get_tokenizer, device, n_mel_channels, n_fft, hop_length, win_length, target_sample_rate, ode_method

class ModelManager:
    def __init__(self, models_dir: str = "src/api/config/models"):
        self.model_configs = load_model_configs(models_dir)
        self.loaded_models = {}

    def get_model_config(self, group_name: str):
        """
        获取指定模型的配置

        Args:
            model_name: 模型名称

        Returns:
            模型配置字典，如果模型不存在则返回None
        """
        return self.model_configs.get(group_name)

    def get_model(self, group_name: str, model_name: str):
        """
        获取指定模型的实例，如果未加载则加载并缓存

        Args:
            model_name: 模型名称

        Returns:
            模型实例，如果模型不存在或加载失败则返回None
        """
        comprehensive_name=group_name+"."+model_name
        if comprehensive_name in self.loaded_models:
            return self.loaded_models[comprehensive_name]

        model_config = self.get_model_config(group_name)
        if not model_config or (not model_name in model_config["models"]):
            print(f"模型 '{comprehensive_name}' 不存在配置。")
            return None

        try:
            # 加载模型配置
            model_cfg = load_config(model_config["model_cfg_path"])
            
            # 动态导入模型类
            backbone_module_name = f"f5_tts.model.backbones.{model_cfg['model']['backbone'].lower()}" # 假设backbone模块名是小写
            backbone_class_name = model_cfg['model']['backbone']
            backbone_module = importlib.import_module(backbone_module_name)
            model_cls = getattr(backbone_module, backbone_class_name)
            
            model_arc = model_cfg['model']['arch']

            # 加载 tokenizer
            print(f'model_config["vocab_path"] : {model_config["vocab_path"]}')
            vocab_char_map, vocab_size = get_tokenizer(model_config["vocab_path"], model_cfg['model']['tokenizer'])

            print(f'model_config["vocab_path"] : {model_config["vocab_path"]}')
            # 创建模型实例
            model = CFM(
                transformer=model_cls(**model_arc, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
                mel_spec_kwargs=dict(
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    n_mel_channels=n_mel_channels,
                    target_sample_rate=target_sample_rate,
                    mel_spec_type=model_cfg['model']['mel_spec']['mel_spec_type'],
                ),
                odeint_kwargs=dict(
                    method=ode_method,
                ),
                vocab_char_map=vocab_char_map,
            ).to(device)

            print(f'model_config["vocab_path"] : {model_config["vocab_path"]}')
            # 加载模型检查点
            model = load_checkpoint(model, os.path.join(model_config["model_path"],model_name), device)

            # 缓存模型
            self.loaded_models[comprehensive_name] = model
            print(f"模型 '{comprehensive_name}' 加载成功并已缓存。")
            return model

        except Exception as e:
            print(f"加载模型 '{comprehensive_name}' 失败: {e}")
            return None

# 示例用法
if __name__ == '__main__':
    model_manager = ModelManager()
    print("\n可用模型配置:")
    for name, config in model_manager.model_configs.items():
        print(f"- {name}: {config['description']} ({config['language']})")

    # 尝试加载一个模型
    model_name_to_load = "F5TTS_Base_Chinese" # 替换为你创建的中文模型名称
    model_instance = model_manager.get_model(model_name_to_load)

    if model_instance:
        print(f"\n成功获取模型实例: {model_name_to_load}")
        # 在这里可以使用模型实例进行推理
    else:
        print(f"\n无法获取模型实例: {model_name_to_load}")

    model_name_to_load_jp = "F5TTS_Japanese" # 替换为你创建的日语模型名称
    model_instance_jp = model_manager.get_model(model_name_to_load_jp)

    if model_instance_jp:
        print(f"\n成功获取模型实例: {model_name_to_load_jp}")
        # 在这里可以使用模型实例进行推理
    else:
        print(f"\n无法获取模型实例: {model_name_to_load_jp}")
