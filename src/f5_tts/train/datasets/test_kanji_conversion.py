import sys
import os

# 直接导入当前目录下的data_prepare模块
from data_prepare import convert_kanji_to_kana

def test_conversion():
    """
    测试日语汉字转换为平假名的功能
    """
    # 测试文本，包含汉字、平假名和片假名
    test_texts = [
        "絆があれば、なんだって乗り越えられると思うんです！",  # 包含汉字和平假名
        "東京は日本の首都です。",  # 简单的句子，包含汉字
        "私はアニメが好きです。",  # 包含汉字、平假名和片假名
        "こんにちは、元気ですか？",  # 主要是平假名，有少量汉字
    ]
    
    print("测试日语汉字转换为平假名：\n")
    
    for i, text in enumerate(test_texts):
        converted = convert_kanji_to_kana(text)
        print(f"原文 {i+1}: {text}")
        print(f"转换后: {converted}")
        print("-" * 50)

if __name__ == "__main__":
    test_conversion()
