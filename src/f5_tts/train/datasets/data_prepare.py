
import os
import whisper
import argparse
import csv
import json
from importlib.resources import files
from pathlib import Path
import os
import sys
import signal
import subprocess  # For invoking ffprobe
import shutil
import concurrent.futures
import multiprocessing
from contextlib import contextmanager
import torchaudio
from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm
import pykakasi  # 导入pykakasi库用于日语汉字转换

CHUNK_SIZE = 100  # Number of files to process per worker batch
THREAD_NAME_PREFIX = "AudioProcessor"
MAX_WORKERS = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU free
executor = None  # Global executor for cleanup
@contextmanager
def graceful_exit():
    """Context manager for graceful shutdown on signals"""

    def signal_handler(signum, frame):
        print("\nReceived signal to terminate. Cleaning up...")
        if executor is not None:
            print("Shutting down executor...")
            executor.shutdown(wait=False, cancel_futures=True)
        sys.exit(1)

    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        yield
    finally:
        if executor is not None:
            executor.shutdown(wait=False)
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

def whisper_transcribe(audio_path, model_size="large", language="ja"):
    model = whisper.load_model(model_size)  # model size:tiny, base, small, medium, large
    result = model.transcribe(audio_path, language=language, fp16=False)
    return result["text"]
def translate_audio_to_text(audio_dir, language="ja"):
    audio_text_pairs = []
    for file in os.listdir(audio_dir):
        if file.endswith(('.wav', '.mp3')):  
            file_path = os.path.join(audio_dir, file)
            text = whisper_transcribe(file_path, language=language)
            audio_text_pairs.append((file_path, text))
    return audio_text_pairs

def process_audio_file(audio_path, text):
    """Process a single audio file by checking its existence and extracting duration."""
    if not Path(audio_path).exists():
        print(f"audio {audio_path} not found, skipping")
        return None
    try:
        audio_duration = get_audio_duration(audio_path)
        if audio_duration <= 0:
            raise ValueError(f"Duration {audio_duration} is non-positive.")
        return (audio_path, text, audio_duration)
    except Exception as e:
        print(f"Warning: Failed to process {audio_path} due to error: {e}. Skipping corrupt file.")
        return None

def get_audio_duration(audio_path, timeout=5):
    """
    Get the duration of an audio file in seconds using ffmpeg's ffprobe.
    Falls back to torchaudio.load() if ffprobe fails.
    """
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            audio_path,
        ]
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True, timeout=timeout
        )
        duration_str = result.stdout.strip()
        if duration_str:
            return float(duration_str)
        raise ValueError("Empty duration string from ffprobe.")
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, ValueError) as e:
        print(f"Warning: ffprobe failed for {audio_path} with error: {e}. Falling back to torchaudio.")
        try:
            audio, sample_rate = torchaudio.load(audio_path)
            return audio.shape[1] / sample_rate
        except Exception as e:
            raise RuntimeError(f"Both ffprobe and torchaudio failed for {audio_path}: {e}")


def save_prepped_dataset(out_dir, result, duration_list, text_vocab_set):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    print(f"\nSaving to {out_dir} ...")

    # Save dataset with improved batch size for better I/O performance
    raw_arrow_path = out_dir / "raw.arrow"
    with ArrowWriter(path=raw_arrow_path.as_posix(), writer_batch_size=2) as writer:
        for line in tqdm(result, desc="Writing to raw.arrow ..."):
            writer.write(line)

    # Save durations to JSON
    dur_json_path = out_dir / "duration.json"
    with open(dur_json_path.as_posix(), "w", encoding="utf-8") as f:
        json.dump({"duration": duration_list}, f, ensure_ascii=False)

    # Handle vocab file - write only once based on finetune flag
    voca_out_path = out_dir / "vocab.txt"
    with open(voca_out_path.as_posix(), "w") as f:
            for vocab in sorted(text_vocab_set):
                f.write(vocab + "\n")

    dataset_name = out_dir.stem
    print(f"\nFor {dataset_name}, sample count: {len(result)}")
    print(f"For {dataset_name}, vocab size is: {len(text_vocab_set)}")
    print(f"For {dataset_name}, total {sum(duration_list) / 3600:.2f} hours")

def prepare_csv_wavs_dir(input_dir, num_workers=None, language="ja"):
    """
    准备音频文件和文本对，支持多语言
    
    参数:
        input_dir: 输入目录，包含音频文件
        num_workers: 工作线程数
        language: 语言代码，默认为日语 "ja"，也支持中文 "zh"，英语 "en" 等
    """
    audio_path_text_pairs = translate_audio_to_text(input_dir, language=language)

    total_files = len(audio_path_text_pairs)

    # Use provided worker count or calculate optimal number
    worker_count = num_workers if num_workers is not None else min(MAX_WORKERS, total_files)
    print(f"\nProcessing {total_files} audio files using {worker_count} workers...")

    with graceful_exit():
        # Initialize thread pool with optimized settings
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=worker_count, thread_name_prefix=THREAD_NAME_PREFIX
        ) as exec:
            executor = exec
            results = []

            # Process files in chunks for better efficiency
            for i in range(0, len(audio_path_text_pairs), CHUNK_SIZE):
                chunk = audio_path_text_pairs[i : i + CHUNK_SIZE]
                # Submit futures in order
                chunk_futures = [executor.submit(process_audio_file, pair[0], pair[1]) for pair in chunk]

                # Iterate over futures in the original submission order to preserve ordering
                for future in tqdm(
                    chunk_futures,
                    total=len(chunk),
                    desc=f"Processing chunk {i // CHUNK_SIZE + 1}/{(total_files + CHUNK_SIZE - 1) // CHUNK_SIZE}",
                ):
                    try:
                        result = future.result()
                        if result is not None:
                            results.append(result)
                    except Exception as e:
                        print(f"Error processing file: {e}")

            executor = None

    # Filter out failed results
    processed = [res for res in results if res is not None]
    if not processed:
        raise RuntimeError("No valid audio files were processed!")

    # Batch process text conversion
    raw_texts = [item[1] for item in processed]
    
    # 对于日语，将汉字转换为平假名
    if language == "ja":
        print("Converting Japanese kanji to hiragana...")
        raw_texts = [convert_kanji_to_kana(text) for text in raw_texts]
    # 对于中文，可以使用 convert_char_to_pinyin 函数进行转换
    
    # Prepare final results
    sub_result = []
    durations = []
    vocab_set = set()

    for (audio_path, _, duration), text in zip(processed, raw_texts):
        sub_result.append({"audio_path": audio_path, "text": text, "duration": duration})
        durations.append(duration)
        vocab_set.update(list(text))

    return sub_result, durations, vocab_set

def prepare_and_save_set(inp_dir, out_dir, num_workers: int = None, language: str = "ja"):
    """
    准备并保存数据集
    
    参数:
        inp_dir: 输入目录
        out_dir: 输出目录
        num_workers: 工作线程数
        language: 语言代码，默认为日语 "ja"
    """
    sub_result, durations, vocab_set = prepare_csv_wavs_dir(inp_dir, num_workers=num_workers, language=language)
    save_prepped_dataset(out_dir, sub_result, durations, vocab_set)

def cli():
    try:
        # Before processing, check if ffprobe is available.
        if shutil.which("ffprobe") is None:
            print(
                "Warning: ffprobe is not available. Duration extraction will rely on torchaudio (which may be slower)."
            )

        # Usage examples in help text
        parser = argparse.ArgumentParser(
            description="Prepare and save dataset.",
            epilog="""
Examples:
    # For fine-tuning (default):
    python data_prepare.py /input/dataset/path /output/dataset/path
    # With custom worker count:
    python data_prepare.py /input/dataset/path /output/dataset/path --workers 4
            """,
        )
        parser.add_argument("inp_dir", type=str, help="Input directory containing the data.")
        parser.add_argument("out_dir", type=str, help="Output directory to save the prepared data.")
        parser.add_argument("--workers", type=int, help=f"Number of worker threads (default: {MAX_WORKERS})")
        parser.add_argument("--language", type=str, default="ja", help="Language code (default: ja for Japanese)")
        args = parser.parse_args()

        prepare_and_save_set(args.inp_dir, args.out_dir, num_workers=args.workers, language=args.language)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user. Cleaning up...")
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
