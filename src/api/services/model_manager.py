import torch
import importlib
import sys
import os

from src.api.utils.config_parser import load_model_configs, load_config
from f5_tts.model import CFM
from f5_tts.infer.utils_infer import load_checkpoint, get_tokenizer, device, n_mel_channels, n_fft, hop_length, win_length, target_sample_rate, ode_method

class ModelManager:
    """
    模型管理器，负责加载和管理模型配置以及模型实例。
    """
    def __init__(self, models_dir: str = "src/api/config/models"):
        """
        初始化 ModelManager。

        Args:
            models_dir: 存放模型配置文件的目录路径。
        """
        self.model_configs = load_model_configs(models_dir)
        self.loaded_models = {}

    def get_model_config(self, group_name: str):
        """
        获取指定模型组的配置。

        Args:
            group_name: 模型组的名称。

        Returns:
            模型配置字典，如果模型组不存在则返回None。
        """
        return self.model_configs.get(group_name)

    def get_model(self, group_name: str, model_name: str):
        """
        获取指定模型组中特定模型的实例。
        如果模型未加载，则加载并缓存模型实例。

        Args:
            group_name: 模型组的名称。
            model_name: 模型组中具体模型的名称（检查点文件名）。

        Returns:
            模型实例，如果模型组或模型不存在，或加载失败则返回None。
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
    # 示例用法
    model_manager = ModelManager()
    print("\n可用模型配置:")
    for name, config in model_manager.model_configs.items():
        print(f"- {name}: {config.get('description', '无描述')} ({config.get('language', '未知语言')})")
        if 'models' in config:
            print(f"  包含模型文件: {', '.join(config['models'])}")

    # 尝试加载一个模型
    # 假设存在一个模型组 'MyModelGroup' 包含模型文件 'my_model.pt'
    group_name_to_load = "japanese_manbo" # 替换为你实际的模型组名称
    model_name_to_load = "japanese_manbo.safetensors" # 替换为你实际的模型文件名称
    model_instance = model_manager.get_model(group_name_to_load, model_name_to_load)

    if model_instance:
        print(f"\n成功获取模型实例: {group_name_to_load}.{model_name_to_load}")
        # 在这里可以使用模型实例进行推理
    else:
        print(f"\n无法获取模型实例: {group_name_to_load}.{model_name_to_load}")
