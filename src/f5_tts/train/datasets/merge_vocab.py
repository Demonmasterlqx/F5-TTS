#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
词汇库合并工具

该脚本用于合并两个词汇库文件，将第一个词汇库中有但第二个词汇库中没有的词汇
添加到第二个词汇库的末尾，并将结果保存到指定的输出文件中。

用法:
    python merge_vocab.py <vocab_a> <vocab_b> <output>

参数:
    vocab_a: 第一个词汇库文件路径
    vocab_b: 第二个词汇库文件路径
    output: 合并后的输出文件路径
"""

import sys
import os


def read_vocab(file_path):
    """
    读取词汇库文件，返回词汇集合

    参数:
        file_path: 词汇库文件路径

    返回:
        词汇集合(set)
    """
    vocab = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word:  # 忽略空行
                    vocab.add(word)
        return vocab
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        sys.exit(1)


def read_vocab_with_order(file_path):
    """
    读取词汇库文件，保持原始顺序

    参数:
        file_path: 词汇库文件路径

    返回:
        词汇列表(list)，保持原始顺序
    """
    vocab = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word:  # 忽略空行
                    vocab.append(word)
        return vocab
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        sys.exit(1)


def merge_vocab(vocab_a_path, vocab_b_path, output_path):
    """
    合并词汇库

    将vocab_a中有但vocab_b中没有的词汇添加到vocab_b的末尾，
    并将结果保存到output_path

    参数:
        vocab_a_path: 第一个词汇库文件路径
        vocab_b_path: 第二个词汇库文件路径
        output_path: 合并后的输出文件路径
    """
    # 读取词汇库
    vocab_a = read_vocab(vocab_a_path)
    vocab_b = read_vocab(vocab_b_path)
    vocab_b_ordered = read_vocab_with_order(vocab_b_path)
    
    # 找出vocab_a中有但vocab_b中没有的词汇
    new_words = vocab_a - vocab_b
    
    # 如果没有新词汇，直接复制vocab_b到output
    if not new_words:
        print(f"没有新词汇需要添加，直接复制 {vocab_b_path} 到 {output_path}")
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(vocab_b_path, 'r', encoding='utf-8') as src, \
                 open(output_path, 'w', encoding='utf-8') as dst:
                dst.write(src.read())
            return
        except Exception as e:
            print(f"复制文件时出错: {e}")
            sys.exit(1)
    
    # 将新词汇添加到vocab_b的末尾
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # 写入原有vocab_b的内容
            for word in vocab_b_ordered:
                f.write(f"{word}\n")
            
            # 写入新词汇
            for word in sorted(new_words):  # 对新词汇排序，使结果更加稳定
                f.write(f"{word}\n")
        
        print(f"成功合并词汇库，共添加了 {len(new_words)} 个新词汇")
        print(f"结果已保存到 {output_path}")
    
    except Exception as e:
        print(f"写入输出文件 {output_path} 时出错: {e}")
        sys.exit(1)


def main():
    """主函数"""
    # 检查命令行参数
    if len(sys.argv) != 4:
        print(f"用法: {sys.argv[0]} <vocab_a> <vocab_b> <output>")
        sys.exit(1)
    
    vocab_a_path = sys.argv[1]
    vocab_b_path = sys.argv[2]
    output_path = sys.argv[3]
    
    # 检查输入文件是否存在
    if not os.path.isfile(vocab_a_path):
        print(f"错误: 文件 {vocab_a_path} 不存在")
        sys.exit(1)
    
    if not os.path.isfile(vocab_b_path):
        print(f"错误: 文件 {vocab_b_path} 不存在")
        sys.exit(1)
    
    # 合并词汇库
    merge_vocab(vocab_a_path, vocab_b_path, output_path)


if __name__ == "__main__":
    main()
