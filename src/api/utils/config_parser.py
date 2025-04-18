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

def load_model_configs(models_dir: str = "config/models"):
    """
    加载所有模型配置文件

    Args:
        models_dir: 存放模型配置文件的目录

    Returns:
        一个字典，key为模型名称，value为模型配置字典
    """
    model_configs = {}
    models_path = Path(models_dir)
    if not models_path.exists():
        print(f"模型配置目录不存在: {models_dir}")
        return model_configs

    for config_file in models_path.glob("*.yaml"):
        try:
            config = load_config(str(config_file))
            if "name" in config:
                model_configs[config["name"]] = config
            else:
                print(f"警告: 模型配置文件 {config_file} 缺少 'name' 字段，将被忽略。")
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
