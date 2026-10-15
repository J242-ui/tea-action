"""YOLOv8 茶叶病害检测训练脚本（适配 2GB 显存/纯 CPU 轻薄本）。

两档：
  --quick  快速档（默认）：subset 400 张 / imgsz 320 / epochs 25 / batch 8
           目的：快速产出一个「可演示」的权重（精度有限，够走通演示链路）
  --full   完整档：全量 7260 张 / imgsz 640 / epochs 100
           目的：复现论文级效果（需要数小时，建议夜间后台跑）

用法（先下载数据，需网络）：
  python training/download_data.py
  python training/train_yolo.py --quick
  python training/train_yolo.py --full

产出：data/models/tea_yolo.pt（best.pt 的副本），供 src/vision/detector.py 加载。
"""
from __future__ import annotations

import os

# Windows 下 numpy/torch 等库可能重复链接 OpenMP 运行时（libiomp5md.dll），
# 必须在 import torch 之前设置，否则训练崩溃（OMP: Error #15）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import re
import shutil
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATASETS = BASE / "training" / "datasets" / "tea_leaf_disease"
MODEL_OUT = BASE / "data" / "models" / "tea_yolo.pt"

QUICK = dict(subset=400, imgsz=320, epochs=25, batch=8, workers=2)
FULL = dict(subset=None, imgsz=640, epochs=100, batch=16, workers=4)


def find_data_yaml(root: Path) -> Path | None:
    """递归定位 data.yaml（兼容压缩包内含顶层目录的情况）。"""
    for p in sorted(root.rglob("data.yaml")):
        return p
    return None


def locate_split(root: Path, split: str) -> Path | None:
    """定位某个 split 目录（兼容 images/ 与直接平铺两种结构）。"""
    for cand in (root / split, root / "images" / split):
        if cand.exists():
            return cand
    return None


def list_images(split_dir: Path) -> list[Path]:
    """列出 split 目录下所有图像文件（兼容 split/images/ 平铺结构）。"""
    d = split_dir / "images" if (split_dir / "images").exists() else split_dir
    return sorted(p for p in d.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def check_deps() -> None:
    try:
        import torch  # noqa: F401
        import ultralytics  # noqa: F401
    except Exception as e:  # noqa: BLE001
        print("[错误] 需要 torch + ultralytics，请先安装（需网络）：")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
        print("  pip install ultralytics opencv-python")
        print(f"  当前错误：{e}")
        sys.exit(1)


def normalize_yaml(src_yaml: Path, out_yaml: Path, root_dir: Path) -> Path:
    """把数据集 data.yaml 规范化为「绝对 path + 相对 train/val/test」形式。

    figshare 原 yaml 的 train 为 `../train/images`，相对路径解析会落空；
    这里固定为绝对 path，子路径统一为 train/images 等，保证 ultralytics 可解析。
    """
    text = src_yaml.read_text(encoding="utf-8", errors="ignore")
    names_m = re.search(r"(?ms)^names:.*$", text)
    nc_m = re.search(r"^nc:\s*\d+.*$", text, re.MULTILINE)
    body = "\n".join(x.group(0) for x in (nc_m, names_m) if x)
    new = (
        f"path: {root_dir.as_posix()}\n"
        f"train: train/images\nval: valid/images\ntest: test/images\n\n"
        f"{body}\n"
    )
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(new, encoding="utf-8")
    return out_yaml


def prepare_dataset(subset: int | None) -> Path:
    """生成（或复用）训练用的规范化 data.yaml；quick 档额外生成 subset 数据目录。"""
    data_yaml = find_data_yaml(DATASETS)
    if data_yaml is None:
        print(f"[错误] 未找到数据集：{DATASETS}")
        print("       请先运行：python training/download_data.py（需要网络）")
        sys.exit(1)
    data_root = data_yaml.parent  # 数据集根目录（含 train/valid/test）

    if subset is None:
        norm = DATASETS / "data_norm.yaml"
        if not norm.exists():
            print(f"[准备] 规范化 data.yaml → {norm}")
            normalize_yaml(data_yaml, norm, data_root)
        else:
            print(f"[复用] 规范化 data.yaml：{norm}")
        return norm

    # quick 档：从 train/valid 各抽子集，保留 images/ + labels/ 相对结构
    sub_dir = DATASETS.parent / "tea_leaf_disease_quick"
    sub_yaml = sub_dir / "data.yaml"
    if sub_yaml.exists():
        print(f"[复用] 快速档数据集已存在：{sub_dir}")
        return sub_yaml

    import random
    rng = random.Random(42)
    for split, take in (("train", subset), ("valid", max(40, subset // 10))):
        src = locate_split(data_root, split)
        if src is None:
            print(f"[警告] 数据集缺少 {split}/ 分片，跳过")
            continue
        imgs = list_images(src)
        if not imgs:
            print(f"[错误] {split}/ 下未找到图像（结构不识别），中止", file=sys.stderr)
            sys.exit(1)
        chosen = rng.sample(imgs, min(take, len(imgs)))
        dst_imgs = sub_dir / split / "images"
        dst_labels = sub_dir / split / "labels"
        dst_imgs.mkdir(parents=True, exist_ok=True)
        label_src_dir = src / "labels" if (src / "labels").exists() else None
        for img in chosen:
            shutil.copy(img, dst_imgs / img.name)
            if label_src_dir is not None:
                label_src = label_src_dir / (img.stem + ".txt")
                if label_src.exists():
                    dst_labels.mkdir(parents=True, exist_ok=True)
                    shutil.copy(label_src, dst_labels / label_src.name)
    normalize_yaml(data_yaml, sub_yaml, sub_dir)
    print(f"[生成] 快速档数据集：{sub_dir}")
    return sub_yaml


def train(cfg: dict) -> Path:
    from ultralytics import YOLO

    device = "cpu"  # 无 NVIDIA 显卡时固定 CPU（2GB 集显无需 CUDA）
    data_yaml = prepare_dataset(cfg["subset"])
    print(f"[训练] YOLOv8n | {cfg}")
    print(f"        数据: {data_yaml}")
    model = YOLO("yolov8n.pt")  # 首次运行自动下载预训练权重（需网络）
    model.train(
        data=str(data_yaml),
        epochs=cfg["epochs"],
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        workers=cfg["workers"],
        device=device,
        project=str(BASE / "training" / "runs"),
        name="tea_yolo",
        exist_ok=True,
        patience=10 if cfg["subset"] else 30,
    )
    # 新版 ultralytics：训练目录从 model.trainer 取，不依赖 train() 返回值
    save_dir = Path(getattr(model.trainer, "save_dir", None) or model.trainer.args.project)
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        print(f"[错误] 训练结束但未找到 best.pt（{best}）", file=sys.stderr)
        sys.exit(1)
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, MODEL_OUT)
    print(f"[完成] 模型已保存：{MODEL_OUT}")
    return MODEL_OUT


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOv8 茶叶病害检测训练")
    parser.add_argument("--quick", action="store_true", help="快速档（默认）")
    parser.add_argument("--full", action="store_true", help="完整档（全量数据，耗时数小时）")
    args = parser.parse_args()

    check_deps()
    cfg = FULL if args.full else QUICK
    out = train(cfg)

    # 提示：训练完成后在服务端热加载模型
    print()
    print("后续步骤：")
    print("  1. 重启 WS 服务，或在 Streamlit「智能执行」页查看模型状态")
    print("  2. src/vision/detector.py 会自动加载新权重；也可调用 detector.reload() 热加载")
    print(f"  3. 模型文件：{out}")


if __name__ == "__main__":
    main()
