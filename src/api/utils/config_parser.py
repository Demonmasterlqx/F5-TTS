import yaml
from pathlib import Path

def load_config(config_path: str):
    """
    加载YAML配置文件

    Args:
        config_path: 配置文件的路径

    Returns:
        解析后的配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def load_model_configs(models_dir: str = "src/api/config/models"):
    """
    加载所有模型配置文件

    Args:
        models_dir: 存放模型配置文件的目录

    Returns:
        一个字典，key为模型名称，value为模型配置字典
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
