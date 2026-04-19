from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

# =========================
# 0) yolo26 仓库路径
# =========================
REPO_ROOT = Path(r"E:\repositories\ultralytics_yolo26")
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from ultralytics import YOLO  # noqa: E402


# =========================
# 1) 用户配置
# =========================
RUNS_ROOT = Path(r"E:\Large Files\UESTC4006P Individual Project (2025-26)\要使用的训练结果汇总")
SOURCE_DIR = Path(r"E:\Large Files\UESTC4006P Individual Project (2025-26)\test\seg")
OUTPUT_ROOT = RUNS_ROOT / "_predict_exports" / "seg_direct_yolo26"

# 只处理名字里带 26 的 seg 目录；若填写白名单，则只跑白名单
RUN_NAME_KEYWORD = "26"
RUN_NAME_WHITELIST: list[str] = [
    # "train_seg_26n",
    # "train_seg_26m",
    # "train_seg_26m_ca_neck",
]

DEVICE = 0
IMGSZ = 1024
CONF = 0.25
IOU = 0.50
MAX_IMAGES = 0   # 0 表示不限制
SAVE_MASK = True
SAVE_TXT = False
SHOW_PROGRESS = True

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def is_target_run(run_dir: Path) -> bool:
    if not run_dir.is_dir():
        return False
    name = run_dir.name
    if not name.startswith("train_seg_"):
        return False
    if RUN_NAME_WHITELIST:
        return name in RUN_NAME_WHITELIST
    return RUN_NAME_KEYWORD in name


def iter_images(folder: Path) -> Iterable[Path]:
    imgs = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    imgs = sorted(imgs)
    if MAX_IMAGES and MAX_IMAGES > 0:
        imgs = imgs[:MAX_IMAGES]
    return imgs


def find_best_weight(run_dir: Path) -> Path:
    pt = run_dir / "weights" / "best.pt"
    if not pt.exists():
        raise FileNotFoundError(f"缺少 best.pt: {pt}")
    return pt


def union_mask_from_result(result, h: int, w: int, thr: float = 0.5) -> np.ndarray:
    if result.masks is None:
        return np.zeros((h, w), dtype=np.uint8)
    m = result.masks.data
    if m is None or len(m) == 0:
        return np.zeros((h, w), dtype=np.uint8)
    m = m.detach().cpu().numpy()
    union = np.max(m, axis=0)
    mask = (union > thr).astype(np.uint8) * 255
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask


def overlay_mask_red(img_bgr: np.ndarray, mask_u8: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    out = img_bgr.copy()
    red = np.zeros_like(img_bgr)
    red[:, :, 2] = 255
    m = mask_u8 > 0
    out[m] = (img_bgr[m] * (1 - alpha) + red[m] * alpha).astype(np.uint8)
    return out


def main() -> None:
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"source 不存在: {SOURCE_DIR}")
    ensure_dir(OUTPUT_ROOT)

    run_dirs = sorted([p for p in RUNS_ROOT.iterdir() if is_target_run(p)])
    if not run_dirs:
        raise RuntimeError("没有找到任何 yolo26 seg 训练目录，请检查 RUNS_ROOT / RUN_NAME_WHITELIST。")

    images = list(iter_images(SOURCE_DIR))
    if not images:
        raise RuntimeError(f"在 {SOURCE_DIR} 下没有找到图片。")

    print("将处理以下 yolo26 seg 模型:")
    for p in run_dirs:
        print(" -", p.name)

    for run_dir in run_dirs:
        weight = find_best_weight(run_dir)
        save_dir = OUTPUT_ROOT / run_dir.name
        ensure_dir(save_dir)

        print(f"\n{'=' * 100}")
        print(f"[开始] {run_dir.name}")
        print(f"权重: {weight}")
        print(f"输出: {save_dir}")

        model = YOLO(str(weight))
        t0_all = time.perf_counter()

        for idx, img_path in enumerate(images, start=1):
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[WARN] 无法读取: {img_path}")
                continue
            h, w = img.shape[:2]

            t0 = time.perf_counter()
            result = model.predict(
                source=img,
                imgsz=IMGSZ,
                conf=CONF,
                iou=IOU,
                device=DEVICE,
                verbose=False,
                save=False,
                save_txt=SAVE_TXT,
            )[0]
            dt = time.perf_counter() - t0

            stem = img_path.stem
            plotted = result.plot()
            cv2.imwrite(str(save_dir / f"{stem}__overlay.jpg"), plotted)

            if SAVE_MASK:
                mask = union_mask_from_result(result, h, w, thr=0.5)
                cv2.imwrite(str(save_dir / f"{stem}__mask.png"), mask)
                ov2 = overlay_mask_red(img, mask, alpha=0.45)
                cv2.imwrite(str(save_dir / f"{stem}__mask_overlay.jpg"), ov2)

            if SHOW_PROGRESS:
                print(f"[{run_dir.name}] {idx}/{len(images)} {img_path.name}  {dt*1000:.1f} ms")

        total_dt = time.perf_counter() - t0_all
        with open(save_dir / "RUN_INFO.txt", "w", encoding="utf-8") as f:
            f.write("YOLO26 direct segmentation batch inference\n")
            f.write(f"repo={REPO_ROOT}\n")
            f.write(f"run_dir={run_dir}\n")
            f.write(f"weight={weight}\n")
            f.write(f"source={SOURCE_DIR}\n")
            f.write(f"imgsz={IMGSZ}, conf={CONF}, iou={IOU}, device={DEVICE}\n")
            f.write(f"images={len(images)}, total_time_s={total_dt:.3f}\n")

        print(f"[完成] {run_dir.name}  total={total_dt:.2f}s")

    print("\n全部完成。")
    print(f"输出根目录: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
