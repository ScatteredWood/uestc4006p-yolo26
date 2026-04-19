from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import torch

# =============================================================================
# 0) 配置区
# =============================================================================

# yolo26 仓库根目录：请按你的实际路径修改
REPO_ROOT = Path(r"E:\repositories\ultralytics_yolo26")

# 训练结果根目录（里面是 train_det_26n、train_seg_26m 等）
RUNS_ROOT = Path(
    r"E:\Large Files\UESTC4006P Individual Project (2025-26)\要使用的训练结果汇总"
)

# det 直接推理测试图像目录
SOURCE_DIR = Path(
    r"E:\Large Files\UESTC4006P Individual Project (2025-26)\test\det"
)

# 导出目录
EXPORT_ROOT = RUNS_ROOT / "_predict_exports" / "det_direct_yolo26"

# 只处理名字里带 26 的 det 目录
RUN_NAME_KEYWORD = "26"

# 如果你只想跑指定几个 det 模型，就在这里填白名单；留空表示自动扫描所有 train_det_* 且包含 26 的目录
RUN_NAME_WHITELIST: list[str] = [
    # "train_det_26n",
    # "train_det_26s",
]

# 推理参数
DEVICE: int | str = 0
IMGSZ = 1024
CONF = 0.25
IOU = 0.50
MAX_DET = 300
LINE_WIDTH = 2
SAVE_TXT = True
SAVE_CONF = True
SAVE_CROP = False
VERBOSE = True
HALF = torch.cuda.is_available() and DEVICE != "cpu"

# 图像扩展名
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# =============================================================================
# 1) 强制使用 yolo26 仓库代码
# =============================================================================

sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from ultralytics import YOLO  # noqa: E402


# =============================================================================
# 2) 工具函数
# =============================================================================

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def is_target_run(run_dir: Path) -> bool:
    name = run_dir.name
    if not run_dir.is_dir():
        return False
    if not name.startswith("train_det_"):
        return False

    if RUN_NAME_WHITELIST:
        return name in RUN_NAME_WHITELIST

    return RUN_NAME_KEYWORD in name


def find_best_weight(run_dir: Path) -> Path | None:
    p = run_dir / "weights" / "best.pt"
    return p if p.exists() else None


def collect_images(src_dir: Path) -> list[Path]:
    if not src_dir.exists():
        raise FileNotFoundError(f"源目录不存在：{src_dir}")
    if not src_dir.is_dir():
        raise NotADirectoryError(f"源路径不是文件夹：{src_dir}")

    images = sorted(
        [p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    )
    if not images:
        raise RuntimeError(f"在 {src_dir} 下没有找到图像文件。")
    return images


def write_run_info(run_dir: Path, run_name: str, weight_path: Path) -> None:
    info_path = run_dir / "RUN_INFO.txt"
    with open(info_path, "w", encoding="utf-8") as f:
        f.write("UESTC4006P - YOLO26 Direct DET Batch Prediction\n")
        f.write("====================================================\n")
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"REPO_ROOT: {REPO_ROOT}\n")
        f.write(f"RUN_NAME: {run_name}\n")
        f.write(f"WEIGHT: {weight_path}\n")
        f.write(f"SOURCE_DIR: {SOURCE_DIR}\n")
        f.write("\n[Predict Params]\n")
        f.write(f"device={DEVICE}\n")
        f.write(f"imgsz={IMGSZ}\n")
        f.write(f"conf={CONF}\n")
        f.write(f"iou={IOU}\n")
        f.write(f"max_det={MAX_DET}\n")
        f.write(f"line_width={LINE_WIDTH}\n")
        f.write(f"half={HALF}\n")
        f.write(f"save_txt={SAVE_TXT}\n")
        f.write(f"save_conf={SAVE_CONF}\n")
        f.write(f"save_crop={SAVE_CROP}\n")
        f.write("====================================================\n")


def run_one_model(run_dir: Path) -> None:
    run_name = run_dir.name
    weight_path = find_best_weight(run_dir)
    if weight_path is None:
        raise FileNotFoundError(f"缺少 best.pt：{run_dir / 'weights' / 'best.pt'}")

    out_dir = EXPORT_ROOT / run_name
    ensure_dir(out_dir)
    write_run_info(out_dir, run_name, weight_path)

    print(f"\n{'=' * 100}")
    print(f"[开始] {run_name}")
    print(f"权重: {weight_path}")
    print(f"输入: {SOURCE_DIR}")
    print(f"输出: {out_dir}")

    model = YOLO(str(weight_path))
    model.predict(
        source=str(SOURCE_DIR),
        device=DEVICE,
        imgsz=IMGSZ,
        conf=CONF,
        iou=IOU,
        max_det=MAX_DET,
        line_width=LINE_WIDTH,
        half=HALF,
        save=True,
        save_txt=SAVE_TXT,
        save_conf=SAVE_CONF,
        save_crop=SAVE_CROP,
        project=str(EXPORT_ROOT),
        name=run_name,
        exist_ok=True,
        verbose=VERBOSE,
    )

    print(f"[完成] {run_name}")


# =============================================================================
# 3) 主流程
# =============================================================================

def main() -> None:
    ensure_dir(EXPORT_ROOT)
    _ = collect_images(SOURCE_DIR)

    run_dirs = sorted([p for p in RUNS_ROOT.iterdir() if is_target_run(p)])
    if not run_dirs:
        raise RuntimeError("没有找到任何 yolo26 det 训练结果目录，请检查 RUNS_ROOT 或 RUN_NAME_WHITELIST。")

    print("当前 yolo26 仓库：", REPO_ROOT)
    print("训练结果根目录：", RUNS_ROOT)
    print("测试图像目录：", SOURCE_DIR)
    print("导出目录：", EXPORT_ROOT)
    print("\n将处理以下 yolo26 det 目录：")
    for p in run_dirs:
        print(" -", p.name)

    failed: list[tuple[str, str]] = []

    for run_dir in run_dirs:
        try:
            run_one_model(run_dir)
        except Exception as e:
            failed.append((run_dir.name, str(e)))
            print(f"[失败] {run_dir.name}: {e}")

    if failed:
        failed_path = EXPORT_ROOT / "FAILED_RUNS.txt"
        with open(failed_path, "w", encoding="utf-8") as f:
            for name, err in failed:
                f.write(f"{name}\t{err}\n")
        print("\n以下模型失败：")
        for name, err in failed:
            print(f" - {name}: {err}")
        print("失败记录已写入：", failed_path)

    print("\n全部结束。")


if __name__ == "__main__":
    main()
