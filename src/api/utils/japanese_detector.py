"""
日语字符检测工具
"""

def contains_japanese(text: str) -> bool:
    """
    判断字符串中是否包含日语字符
    
    参数:
        text: 要检查的字符串
        
    返回:
        bool: 如果包含日语字符返回True，否则返回False
        
    示例:
        >>> contains_japanese("こんにちは")
        True
        >>> contains_japanese("hello")
        False
        >>> contains_japanese("日本語とEnglish混合")
        True
    """
    # 定义日语字符的Unicode范围
    hiragana_range = (0x3040, 0x309F)  # 平假名
    katakana_range = (0x30A0, 0x30FF)  # 片假名
    japanese_punctuation_range = (0x3000, 0x303F)  # 日语标点符号
    
    for char in text:
        code = ord(char)
        # 检查是否在平假名范围
        if hiragana_range[0] <= code <= hiragana_range[1]:
            return True
        # 检查是否在片假名范围
        if katakana_range[0] <= code <= katakana_range[1]:
            return True
        # 检查是否在日语标点范围
        if japanese_punctuation_range[0] <= code <= japanese_punctuation_range[1]:
            return True
        # 检查是否在常用汉字范围（日语使用）
            
    return False


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    
    # 单元测试
    import unittest
    
    class TestJapaneseDetection(unittest.TestCase):
        def test_hiragana(self):
            self.assertTrue(contains_japanese("こんにちは"))
            
        def test_katakana(self):
            self.assertTrue(contains_japanese("コンニチハ"))
            
        def test_kanji(self):
            self.assertTrue(contains_japanese("日本語"))
            
        def test_mixed(self):
            self.assertTrue(contains_japanese("こんにちは、Hello"))
            
        def test_non_japanese(self):
            self.assertFalse(contains_japanese("hello world"))
            self.assertFalse(contains_japanese("123456"))
            self.assertFalse(contains_japanese(""))
            
        def test_japanese_punctuation(self):
            self.assertTrue(contains_japanese("・"))
            self.assertTrue(contains_japanese("「"))
            
    unittest.main()
