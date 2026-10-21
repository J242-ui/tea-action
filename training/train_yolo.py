"""YOLOv8 茶叶病害检测训练脚本（适配 2GB 显存 / 纯 CPU 轻薄本）。



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
# 无显示环境：matplotlib 使用 Agg 后端，只保存图片不弹窗
import matplotlib
matplotlib.use("Agg")

import argparse
import random
import re
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 中文字体（Windows 微软雅黑），避免图内中文变方框
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(__file__).resolve().parent.parent
DATASETS = BASE / "training" / "datasets" / "tea_leaf_disease"
MODEL_OUT = BASE / "data" / "models" / "tea_yolo.pt"
RUNS = BASE / "training" / "runs"

QUICK = dict(subset=400, imgsz=320, epochs=25, batch=8, workers=2)
FULL = dict(subset=None, imgsz=640, epochs=100, batch=16, workers=4)

SEED = 42


# ============================================================
# 数据准备工具（兼容压缩包内含顶层目录 / 平铺结构等多种布局）
# ============================================================
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
    """【前置检查】确认 torch + ultralytics 已安装，缺失则给出安装指引。"""
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


def yaml_ok(yaml_path: Path) -> bool:
    """校验 data.yaml 的 path 字段指向的目录真实存在。

    项目目录改过名（旧名「项目文件」），历史生成的 data.yaml 可能残留旧绝对路径，
    直接复用会让 ultralytics 找不到数据而报错，因此复用前必须校验。
    """
    try:
        import yaml  # ultralytics 依赖自带

        with open(yaml_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        root = Path(str(cfg.get("path", "")))
        return root.is_dir()
    except Exception:  # noqa: BLE001
        return False


def prepare_dataset(subset: int | None) -> Path:
    """生成（或复用）训练用的规范化 data.yaml；quick 档额外生成 subset 数据目录。"""
    data_yaml = find_data_yaml(DATASETS)
    if data_yaml is None:
        print(f"[错误] 未找到数据集：{DATASETS}")
        print("       请先运行：python training/download_data.py（需要网络）")
        sys.exit(1)
    data_root = data_yaml.parent  # 数据集根目录（含 train/valid/test）

    if subset is None:
        # 完整档：直接复用规范化 yaml（若旧 yaml 的 path 失效则重新生成）
        norm = DATASETS / "data_norm.yaml"
        if norm.exists() and yaml_ok(norm):
            print(f"[复用] 规范化 data.yaml：{norm}")
        else:
            print(f"[准备] 规范化 data.yaml → {norm}")
            normalize_yaml(data_yaml, norm, data_root)
        return norm

    # quick 档：从 train/valid 各抽子集，保留 images/ + labels/ 相对结构
    sub_dir = DATASETS.parent / "tea_leaf_disease_quick"
    sub_yaml = sub_dir / "data.yaml"
    if sub_yaml.exists() and yaml_ok(sub_yaml):
        print(f"[复用] 快速档数据集已存在：{sub_dir}")
        return sub_yaml

    if sub_yaml.exists():
        # 数据集文件还在，仅旧 yaml 的 path 失效 → 只重建 yaml，不重新拷贝
        print(f"[修复] 快速档 data.yaml 指向旧路径，重新生成：{sub_yaml}")
        normalize_yaml(data_yaml, sub_yaml, sub_dir)
        return sub_yaml

    rng = random.Random(SEED)
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


# ============================================================
# 1. 设置随机种子（保证每次训练结果可复现）
# ============================================================
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"[1/11] 随机种子已固定：{seed}")


# ============================================================
# 2. 设置运行设备（有 NVIDIA 显卡自动用 CUDA，否则用 CPU）
# ============================================================
def detect_device() -> str:
    import torch

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"[2/11] 运行设备：CUDA（显卡：{name}）")
        return "cuda"
    print("[2/11] 运行设备：CPU（未检测到 NVIDIA 显卡；核显/无独显机器自动走 CPU，稳定但较慢）")
    return "cpu"


# ============================================================
# 3. 数据集准备与统计（打印类别、每类样本数、图像总量）
# ============================================================
def inspect_dataset(data_yaml: Path) -> tuple[list[str], dict[str, int]]:
    import yaml

    with open(data_yaml, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    names = list(cfg.get("names", []))
    root = Path(str(cfg.get("path", "")))

    # 逐类统计标签出现次数（读 labels/*.txt 的首列类别号）
    class_count = {n: 0 for n in names}
    img_total = 0
    for split in ("train", "valid", "test"):
        sd = locate_split(root, split)
        if sd is None:
            continue
        imgs = list_images(sd)
        img_total += len(imgs)
        label_dir = sd / "labels" if (sd / "labels").exists() else None
        if label_dir is None:
            continue
        for lb in label_dir.glob("*.txt"):
            try:
                first = lb.read_text(encoding="utf-8", errors="ignore").split()
                if first:
                    idx = int(float(first[0]))
                    if 0 <= idx < len(names):
                        class_count[names[idx]] += 1
            except (ValueError, OSError):
                continue
    return names, class_count, img_total


def show_samples(data_yaml: Path, out_png: Path) -> None:
    """【样本可视化】从训练集抽 9 张原图网格展示，保存 PNG。"""
    import yaml

    with open(data_yaml, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = Path(str(cfg.get("path", "")))
    tr = locate_split(root, "train")
    if tr is None:
        print("        [跳过] 未找到训练图像目录")
        return
    imgs = list_images(tr)
    if not imgs:
        print("        [跳过] 训练集为空")
        return
    rng = random.Random(SEED)
    chosen = rng.sample(imgs, min(9, len(imgs)))
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for ax, p in zip(axes.flat, chosen):
        img = plt.imread(p)
        ax.imshow(img)
        ax.set_title(p.stem[:14], fontsize=8)
        ax.axis("off")
    fig.suptitle("茶叶病害训练样本（原始图像）", fontsize=14)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"        样本图已保存：{out_png}")


# ============================================================
# 8. 训练曲线可视化（读取 ultralytics 自动生成的 results.csv）
# ============================================================
def plot_curves(csv_path: Path, out_dir: Path) -> None:
    import pandas as pd

    if not csv_path.exists():
        print("        [跳过] 未找到 results.csv，无法绘制曲线")
        return
    df = pd.read_csv(csv_path)
    cols = list(df.columns)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 图1：损失曲线（训练/验证 box loss；若存在 cls/dfl 一并画出）
    loss_pairs = [
        ("train/box_loss", "val/box_loss"),
        ("train/cls_loss", "val/cls_loss"),
        ("train/dfl_loss", "val/dfl_loss"),
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    for tr_c, va_c in loss_pairs:
        if tr_c in cols and va_c in cols:
            ax.plot(df[tr_c], label=f"训练 {tr_c.split('/')[-1]}")
            ax.plot(df[va_c], label=f"验证 {va_c.split('/')[-1]}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("训练/验证损失曲线")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "curves_loss.png", dpi=120)
    plt.close(fig)

    # 图2：评估指标曲线（精确率/召回率/mAP50/mAP50-95）
    metric_cols = [
        ("metrics/precision(B)", "精确率 Precision"),
        ("metrics/recall(B)", "召回率 Recall"),
        ("metrics/mAP50(B)", "mAP50"),
        ("metrics/mAP50-95(B)", "mAP50-95"),
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    for c, label in metric_cols:
        if c in cols:
            ax.plot(df[c], label=label, linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("训练指标曲线")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "curves_metrics.png", dpi=120)
    plt.close(fig)
    print("        训练曲线已保存：curves_loss.png / curves_metrics.png")


# ============================================================
# 9. 验证评估：混淆矩阵 + 逐类别分类报告（mAP50-95）
# ============================================================
def evaluate(model, save_dir: Path) -> None:
    print("        [9/11] 在验证集上评估...")
    metrics = model.val(plots=True, save_json=False)  # plots=True 生成混淆矩阵图

    maps = list(metrics.box.maps) if metrics.box.maps is not None else []
    names = model.names
    print("        类别                            mAP50-95")
    for i in range(len(names)):
        v = maps[i] if i < len(maps) else float("nan")
        print(f"        {i} {names[i]:<30s} {v:.4f}")
    if maps:
        print(f"        全类别平均 mAP50-95：{sum(x for x in maps if x == x) / len(maps):.4f}")

    # 混淆矩阵图（ultralytics 训练/验证时自动生成）
    for cand in (save_dir / "confusion_matrix.png", save_dir / "confusion_matrix_normalized.png"):
        if cand.exists():
            print(f"        混淆矩阵图：{cand}")
            break
    else:
        print("        [提示] 未找到混淆矩阵图（可在验证参数 plots=True 时生成）")


# ============================================================
# 11. 预测演示：用训练好的权重对验证集抽样预测并保存标注图
# ============================================================
def demo_predict(model_path: Path, data_yaml: Path, device: str, imgsz: int, out_dir: Path) -> None:
    from ultralytics import YOLO

    import yaml

    with open(data_yaml, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = Path(str(cfg.get("path", "")))
    va = locate_split(root, "valid") or locate_split(root, "test")
    if va is None:
        print("        [跳过] 未找到验证集目录")
        return
    imgs = list_images(va)[:6]
    if not imgs:
        print("        [跳过] 验证集为空")
        return
    model = YOLO(str(model_path))
    model.predict(
        source=[str(p) for p in imgs],
        imgsz=imgsz,
        device=device,
        save=True,
        project=str(out_dir),
        name="demo",
        exist_ok=True,
        conf=0.25,
    )
    demo_dir = out_dir / "demo"
    print(f"        预测标注图已保存：{demo_dir}")


# ============================================================
# 训练主流程
# ============================================================
def train(cfg: dict) -> Path:
    import torch
    from ultralytics import YOLO

    # 节1-2：种子 + 设备
    set_seed()
    device = detect_device()

    # 节3：数据集准备与统计
    print("\n[3/11] 准备数据集...")
    data_yaml = prepare_dataset(cfg["subset"])
    names, class_count, img_total = inspect_dataset(data_yaml)
    print(f"        数据文件：{data_yaml}")
    print(f"        类别数：{len(names)}，图像总量：{img_total}")
    for n in names:
        print(f"          - {n}（样本 {class_count[n]}）")

    # 节4：样本可视化
    print("\n[4/11] 可视化数据样本...")
    show_samples(data_yaml, RUNS / "samples.png")

    # 节5：训练配置
    print("\n[5/11] 训练配置：")
    for k, v in cfg.items():
        print(f"        {k} = {v}")
    print(f"        设备 = {device}，随机种子 = {SEED}")

    # 节6：创建模型（首次运行自动下载 yolov8n 预训练权重，需网络）
    print("\n[6/11] 创建 YOLOv8n 模型（首次运行会下载预训练权重）...")
    model = YOLO("yolov8n.pt")
    print(f"        模型：{model.task} / 参数规模：{sum(p.numel() for p in model.model.parameters()):,}")

    # 节7：开始训练
    print("\n[7/11] 开始训练（ultralytics 会自动保存每轮日志与最佳权重）...")
    model.train(
        data=str(data_yaml),
        epochs=cfg["epochs"],
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        workers=cfg["workers"],
        device=device,
        project=str(RUNS),
        name="tea_yolo",
        exist_ok=True,
        patience=10 if cfg["subset"] else 30,
        seed=SEED,
        plots=True,
    )

    # 新版 ultralytics：训练目录从 model.trainer 取，不依赖 train() 返回值
    save_dir = Path(getattr(model.trainer, "save_dir", None) or model.trainer.args.project)
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        print(f"[错误] 训练结束但未找到 best.pt（{best}）", file=sys.stderr)
        sys.exit(1)

    # 节8：训练曲线可视化
    print("\n[8/11] 绘制训练曲线...")
    plot_curves(save_dir / "results.csv", save_dir)

    # 节9：验证评估
    print("\n[9/11] 验证评估（混淆矩阵 + 逐类别分类报告）...")
    evaluate(model, save_dir)

    # 节10：保存模型（拷贝 best.pt 到 data/models，供服务端加载）
    print("\n[10/11] 保存模型...")
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, MODEL_OUT)
    print(f"        模型已保存：{MODEL_OUT}")

    # 节11：预测演示
    print("\n[11/11] 预测演示（对验证集抽样检测）...")
    demo_predict(MODEL_OUT, data_yaml, device, cfg["imgsz"], save_dir)

    return MODEL_OUT


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOv8 茶叶病害检测训练")
    parser.add_argument("--quick", action="store_true", help="快速档（默认）")
    parser.add_argument("--full", action="store_true", help="完整档（全量数据，耗时数小时）")
    args = parser.parse_args()

    print("=" * 60)
    print("   YOLOv8 茶叶病害检测 · 训练脚本")
    print("=" * 60)

    check_deps()
    cfg = FULL if args.full else QUICK
    out = train(cfg)

    print()
    print("=" * 60)
    print("   训练完成！后续步骤：")
    print("   1. 重启 WS 服务，或在 Streamlit「智能执行」页查看模型状态")
    print("   2. src/vision/detector.py 会自动加载新权重；也可调用 detector.reload() 热加载")
    print(f"   3. 模型文件：{out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
