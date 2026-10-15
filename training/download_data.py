"""下载茶叶病害数据集（figshare Tea-Leaf Disease and Pest Detection Dataset v1.0）。

数据集信息：
  - 9,591 张标注图像 / 8 类（Black rot, Brown blight, Leaf rust, Red Spider,
    Tea Mosquito bug, Healthy, White spot, disease）
  - YOLO 格式标注（class_id, cx, cy, w, h），已分 train/valid/test
  - 压缩包约 367 MB，CC BY 4.0
  - 来源: https://figshare.com/articles/dataset/Tea-Leaf_Disease_and_Pest_Detection_Dataset_v1_0_/32253357

用法：
  python training/download_data.py            # 下载并解压到 training/datasets/
  python training/download_data.py --skip-download   # 仅解压已下载的 zip
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import requests

DATASET_URL = "https://figshare.com/ndownloader/articles/32253357/versions/1"
BASE = Path(__file__).resolve().parent
DATASETS = BASE / "datasets"
ZIP_PATH = DATASETS / "tea_leaf_disease_v1.zip"
EXTRACT_DIR = DATASETS / "tea_leaf_disease"
EXPECTED_MB = 367


def download() -> Path:
    DATASETS.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists() and ZIP_PATH.stat().st_size > EXPECTED_MB * 1024 * 1024 * 0.9:
        print(f"[跳过] 已存在 {ZIP_PATH.name}（{ZIP_PATH.stat().st_size / 1e6:.0f} MB）")
        return ZIP_PATH
    print(f"[下载] {DATASET_URL}")
    print(f"       保存到 {ZIP_PATH}（约 {EXPECTED_MB} MB，请保持网络畅通）")
    try:
        with requests.get(DATASET_URL, stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            done = 0
            with open(ZIP_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 512):
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = done * 100 / total
                        print(f"\r  进度 {done / 1e6:.0f}/{total / 1e6:.0f} MB（{pct:.1f}%）", end="")
    except Exception as e:  # noqa: BLE001
        ZIP_PATH.unlink(missing_ok=True)
        print(f"\n[错误] 下载失败：{e}", file=sys.stderr)
        print("       请检查网络连接（figshare 需要可访问外网），然后重试。", file=sys.stderr)
        sys.exit(1)
    print("\n[完成] 下载结束")
    return ZIP_PATH


def extract(zip_path: Path) -> Path:
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    if find_data_yaml(EXTRACT_DIR) is not None:
        print(f"[跳过] 已解压到 {EXTRACT_DIR}")
        return EXTRACT_DIR
    print(f"[解压] {zip_path.name} → {EXTRACT_DIR}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(EXTRACT_DIR)
    print("[完成] 解压结束")
    return EXTRACT_DIR


def find_data_yaml(root: Path) -> Path | None:
    """递归定位 data.yaml（兼容压缩包内含顶层目录的情况）。"""
    for p in sorted(root.rglob("data.yaml")):
        return p
    return None


def show_structure(root: Path) -> None:
    print("\n[结构] 数据集目录：")
    yaml_file = find_data_yaml(root)
    if yaml_file is not None:
        print(f"  - data.yaml 位置：{yaml_file.relative_to(root)}")
        print("  - data.yaml 内容：")
        print("  " + "\n  ".join(yaml_file.read_text(encoding="utf-8", errors="ignore").strip().splitlines()))
    for sub in ("train", "valid", "test"):
        d = root / sub
        if not d.exists():
            # 兼容 zip 内套一层目录的情况
            d = next((p for p in root.iterdir() if p.is_dir() and (p / sub).exists()), None)
            d = d / sub if d else None
        if d is not None and d.exists():
            imgs_dir = d / "images" if (d / "images").exists() else d
            imgs = sum(1 for _ in imgs_dir.glob("*") if _.suffix.lower() in {".jpg", ".jpeg", ".png"})
            labels = (d / "labels").exists()
            print(f"  - {sub}/: {imgs} 张图像（labels 目录: {'有' if labels else '无'}）")


def main() -> None:
    parser = argparse.ArgumentParser(description="下载并解压茶叶病害数据集")
    parser.add_argument("--skip-download", action="store_true", help="跳过下载，仅解压已有 zip")
    args = parser.parse_args()

    zip_path = ZIP_PATH if args.skip_download else download()
    if not zip_path.exists():
        print("[错误] zip 不存在。请先完成下载（或检查网络后重试）。", file=sys.stderr)
        sys.exit(1)
    root = extract(zip_path)
    show_structure(root)


if __name__ == "__main__":
    main()
