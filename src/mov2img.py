import subprocess
import os
from pathlib import Path
import math

# dotenvを使用して.envファイルから設定を読み込み
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / '.env')
except ImportError:
    pass

BASE_DIR = os.environ.get("KAO_BASE_DIR")
INPUT_DIR = Path(BASE_DIR + "/" + os.environ.get("KAO_MOV_DIR"))
OUTPUT_DIR = Path(BASE_DIR + "/" + os.environ.get("KAO_SC_DIR"))

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
FPS = 1
IMAGE_EXT = "jpg"
START_INDEX = 1  # 連番の開始番号

def get_duration(video_path: Path) -> float:
    """動画の長さ（秒）を取得"""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    out = subprocess.check_output(cmd, text=True)
    return float(out.strip())


def extract_frames(video_path: Path, out_dir: Path, start_number: int) -> int:
    """
    フレーム抽出を実行
    戻り値: この動画で生成されたフレーム数
    """
    output_pattern = out_dir / f"%06d.{IMAGE_EXT}"

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(video_path),
        "-vf", f"fps={FPS}",
        "-start_number", str(start_number),
        str(output_pattern),
    ]

    subprocess.run(cmd, check=True)

    # おおよその生成枚数 = 秒数 × FPS
    duration = get_duration(video_path)
    return math.ceil(duration * FPS)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    index = START_INDEX

    for video_path in sorted(INPUT_DIR.iterdir()):
        if video_path.suffix.lower() not in VIDEO_EXTS:
            continue

        try:
            print(f"[PROCESS] {video_path.name} (start={index})")
            count = extract_frames(video_path, OUTPUT_DIR, index)
            index += count
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] {video_path.name}: {e}")

    print(f"[DONE] total images = {index - START_INDEX}")


if __name__ == "__main__":
    main()
