import yaml
from pathlib import Path

def load_config(config_path: str):
    """
    加载YAML配置文件。

    Args:
        config_path: 配置文件的路径。

    Returns:
        解析后的配置字典。
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def load_model_configs(models_dir: str = "src/api/config/models"):
    """
    加载指定目录下所有模型配置文件。

    每个模型配置文件应为YAML格式，包含以下必需字段：
    - name (str): 模型组的名称，作为唯一标识符。
    - language (list[str]): 模型组支持的语言列表。
    - model_path (str): 模型检查点文件 (.pt 或 .safetensors) 或包含多个检查点文件的目录路径。
    - vocab_path (str): 词汇表文件 (.txt) 的路径。
    - model_cfg_path (str): 模型结构配置文件 (.yaml) 的路径。

    Args:
        models_dir: 存放模型配置文件的目录路径。

    Returns:
        一个字典，key为模型组的名称 (name字段)，value为对应的模型配置字典。
        如果模型配置目录不存在或加载配置文件失败，返回空字典。
        如果配置文件缺少必需字段或路径无效，将打印警告并忽略该文件。
    """
    model_configs = {}
    models_path = Path(models_dir)
    needed_key=["name",'language','model_path','vocab_path','model_cfg_path']
    needed_path=['model_path','vocab_path','model_cfg_path']
    if not models_path.exists():
        print(f"模型配置目录不存在: {models_dir}")
        return model_configs

    for config_file in models_path.glob("*.yaml"):
        try:
            config = load_config(str(config_file))
            missing_keys = [key for key in needed_key if key not in config]
            if missing_keys:
                print(f"警告: 模型配置文件 {config_file} 缺少以下字段: {', '.join(missing_keys)}，将被忽略。")
                continue # Skip this config file if keys are missing
            missing_path= [key for key in needed_path if not Path(config[key]).exists()]
            if missing_path:
                print(f"警告: 模型配置文件 {config_file} 以下路径无效: {', '.join(missing_path)}，将被忽略。")
                continue # Skip this config file if keys are missing

            # Check if model_path is a .pt file or a directory
            model_path = Path(config["model_path"])
            # Check if model_path is a .pt file or a directory
            if not (model_path.is_file() and (str(model_path).endswith(".pt") or str(model_path).endswith(".safetensors"))) and not model_path.is_dir():
                print(f"警告: 模型配置文件 {config_file} 中的 'model_path' ('{model_path}') 不是一个 .pt 或者 .safetensors 文件也不是一个目录，将被忽略。")
                continue
            
            if model_path.is_file() and (str(model_path).endswith(".pt") or str(model_path).endswith(".safetensors")):
                model_filename = model_path.name
                config["models"]=[model_filename]
                config["model_path"]=model_path.parent
            elif model_path.is_dir():
                models=[name.name for name in model_path.glob("*.safetensors")]+[name.name for name in model_path.glob("*.pt")]
                config["models"]=models

            model_configs[config["name"]] = config

        except Exception as e:
            print(f"加载模型配置文件 {config_file} 失败: {e}")

    return model_configs

if __name__ == '__main__':
    # 示例用法
    model_configs = load_model_configs()
    print("加载的模型配置:")
    for name, config in model_configs.items():
        print(f"模型名称: {name}")
        print(f"配置: {config}")
        print("-" * 20)
