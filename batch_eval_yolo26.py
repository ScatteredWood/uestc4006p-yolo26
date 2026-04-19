from __future__ import annotations

import json
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml


# =============================================================================
# 0) 先配置你的 yolo26 仓库路径
# =============================================================================
REPO_ROOT = Path(r"E:\repositories\ultralytics_yolo26")

# 强制把 yolo26 仓库放到最前面，确保 import 到的是这个仓库里的代码
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from ultralytics import YOLO  # noqa: E402


# =============================================================================
# 1) 配置区
# =============================================================================

# 训练结果根目录（里面是 train_det_26n、train_seg_26m 等）
RUNS_ROOT = Path(
    r"E:\Large Files\UESTC4006P Individual Project (2025-26)\要使用的训练结果汇总"
)

# 单独给 yolo26 导出一个目录，避免和之前混在一起
EXPORT_ROOT = RUNS_ROOT / "_eval_exports_yolo26"
DATA_YAML_DIR = EXPORT_ROOT / "data_yaml"
RUN_REPORTS_DIR = EXPORT_ROOT / "run_reports"

# det / seg 的验证集图片目录
DET_VAL_IMAGES = Path(
    r"E:\Large Files\UESTC4006P Individual Project (2025-26)\datasets\public\RDD2022_China\yolo_det_D00D10D20D40_seed42_v1\images\val"
)
SEG_VAL_IMAGES = Path(
    r"E:\Large Files\UESTC4006P Individual Project (2025-26)\datasets\custom\crack500_SS305_CSa100_cs4029\crack500_SS305_CSa100_cs4029\images\val"
)

# 统一评估尺寸
DET_IMGSZ = 1024
SEG_IMGSZ = 1024

# 设备
DEVICE: int | str = 0
WORKERS = 4
BATCH = 1
VERBOSE = True
HALF = torch.cuda.is_available() and DEVICE != "cpu"

# 类别名
DET_NAMES = ["D00", "D10", "D20", "D40"]
SEG_NAMES = ["crack"]   # 如果你的 seg 不是单类，这里改成真实类别名

# 只处理名字里带 26 的 run
RUN_NAME_KEYWORD = "26"

# 如果你只想跑指定几个，也可以在这里显式列出；为空表示自动扫描所有包含 26 的目录
RUN_NAME_WHITELIST: list[str] = [
    # "train_det_26n",
    # "train_seg_26n",
    # "train_seg_26m",
    # "train_seg_26m_ca_neck",
]


# =============================================================================
# 2) 工具函数
# =============================================================================

def ensure_dirs() -> None:
    DATA_YAML_DIR.mkdir(parents=True, exist_ok=True)
    RUN_REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def safe_int(x: Any) -> int | None:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def is_yolo26_run(run_dir: Path) -> bool:
    name = run_dir.name
    if not run_dir.is_dir():
        return False
    if not name.startswith("train_"):
        return False

    if RUN_NAME_WHITELIST:
        return name in RUN_NAME_WHITELIST

    return RUN_NAME_KEYWORD in name


def infer_task(run_dir: Path) -> str:
    n = run_dir.name.lower()
    if n.startswith("train_det_"):
        return "detect"
    if n.startswith("train_seg_"):
        return "segment"
    raise ValueError(f"无法从目录名判断任务类型：{run_dir.name}")


def find_best_weight(run_dir: Path) -> Path | None:
    p = run_dir / "weights" / "best.pt"
    return p if p.exists() else None


def load_ckpt_meta(weight_path: Path) -> dict:
    try:
        ckpt = torch.load(weight_path, map_location="cpu")
        if isinstance(ckpt, dict):
            return ckpt
        return {}
    except Exception:
        return {}


def get_best_epoch_from_ckpt(ckpt: dict) -> int | None:
    epoch = ckpt.get("epoch", None)
    if epoch is None:
        return None
    return int(epoch) + 1


def get_weight_size_mb(weight_path: Path) -> float:
    return weight_path.stat().st_size / (1024 ** 2)


def build_val_yaml(images_val_dir: Path, names: list[str], yaml_path: Path) -> None:
    """
    只为本次 val 评估生成临时 YAML。
    train/val/test 都指向 images/val，避免别的划分缺失影响解析。
    """
    dataset_root = images_val_dir.parent.parent
    data = {
        "path": str(dataset_root),
        "train": "images/val",
        "val": "images/val",
        "test": "images/val",
        "names": names,
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def check_image_label_pairs(images_val_dir: Path) -> None:
    if not images_val_dir.exists():
        raise FileNotFoundError(f"缺少图片目录：{images_val_dir}")

    label_val_dir = images_val_dir.parent.parent / "labels" / "val"
    if not label_val_dir.exists():
        raise FileNotFoundError(f"缺少标注目录：{label_val_dir}")

    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    image_files = sorted(
        [p for p in images_val_dir.iterdir() if p.is_file() and p.suffix.lower() in img_exts]
    )

    missing_labels = []
    for img in image_files:
        txt = label_val_dir / f"{img.stem}.txt"
        if not txt.exists():
            missing_labels.append(img.name)

    print(f"\n[检查] {images_val_dir}")
    print(f"图片数量: {len(image_files)}")
    print(f"缺少标注数量: {len(missing_labels)}")

    if missing_labels:
        raise ValueError(f"以下图片缺少对应 txt 标注（仅展示前20个）：{missing_labels[:20]}")


def get_names_map(results, fallback_names: list[str]) -> dict[int, str]:
    names = getattr(results, "names", None)

    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}

    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}

    return {i: str(v) for i, v in enumerate(fallback_names)}


def get_model_info(weight_path: Path, imgsz: int) -> dict:
    """
    获取 params / flops。
    尽量兼容不同版本返回格式。
    """
    out = {
        "layers": None,
        "params": None,
        "grads": None,
        "flops_g": None,
    }

    try:
        model = YOLO(str(weight_path))

        try:
            model.fuse()
        except Exception:
            pass

        try:
            info = model.info(verbose=False, imgsz=imgsz)
            if isinstance(info, (list, tuple)) and len(info) >= 4:
                out["layers"] = safe_int(info[0])
                out["params"] = safe_int(info[1])
                out["grads"] = safe_int(info[2])
                out["flops_g"] = safe_float(info[3])
        except Exception:
            pass

        try:
            inner_model = model.model
            if out["params"] is None:
                out["params"] = int(sum(p.numel() for p in inner_model.parameters()))
            if out["grads"] is None:
                out["grads"] = int(sum(p.numel() for p in inner_model.parameters() if p.requires_grad))
        except Exception:
            pass

    except Exception:
        pass

    return out


def run_val(weight_path: Path, data_yaml: Path, imgsz: int, run_name: str):
    save_dir = RUN_REPORTS_DIR / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weight_path))
    results = model.val(
        data=str(data_yaml),
        split="val",
        imgsz=imgsz,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        half=HALF,
        plots=True,
        verbose=VERBOSE,
        project=str(RUN_REPORTS_DIR),
        name=run_name,
        exist_ok=True,
    )
    return results, save_dir


def export_confusion_matrix(results, save_dir: Path, fallback_names: list[str]) -> None:
    cm_obj = getattr(results, "confusion_matrix", None)
    if cm_obj is None:
        return

    matrix = getattr(cm_obj, "matrix", None)
    if matrix is None:
        return

    try:
        import numpy as np

        mat = np.array(matrix, dtype=float)
        names_map = get_names_map(results, fallback_names)
        n = len(names_map)
        labels = [names_map[i] for i in range(n)]

        if mat.shape[0] == n + 1 and mat.shape[1] == n + 1:
            labels_with_bg = labels + ["background"]
        else:
            labels_with_bg = [f"class_{i}" for i in range(mat.shape[0])]

        raw_df = pd.DataFrame(mat, index=labels_with_bg, columns=labels_with_bg)
        raw_df.to_csv(save_dir / "confusion_matrix_raw.csv", encoding="utf-8-sig")

        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        norm = mat / row_sums
        norm_df = pd.DataFrame(norm, index=labels_with_bg, columns=labels_with_bg)
        norm_df.to_csv(save_dir / "confusion_matrix_normalized.csv", encoding="utf-8-sig")

    except Exception:
        try:
            raw_df = pd.DataFrame(cm_obj.summary(normalize=False, decimals=6))
            raw_df.to_csv(save_dir / "confusion_matrix_raw.csv", index=False, encoding="utf-8-sig")
        except Exception:
            pass

        try:
            norm_df = pd.DataFrame(cm_obj.summary(normalize=True, decimals=6))
            norm_df.to_csv(save_dir / "confusion_matrix_normalized.csv", index=False, encoding="utf-8-sig")
        except Exception:
            pass


def export_per_class_detect(results, save_dir: Path, fallback_names: list[str]) -> None:
    """
    det 每类导出：
    Precision / Recall / F1 / AP50 / AP50-95
    """
    rows: list[dict[str, Any]] = []

    try:
        names_map = get_names_map(results, fallback_names)
        box = results.box

        ap_class_index = list(getattr(box, "ap_class_index", []))
        p_arr = list(getattr(box, "p", []))
        r_arr = list(getattr(box, "r", []))
        f1_arr = list(getattr(box, "f1", []))

        base = {}
        for cls_id, cls_name in names_map.items():
            base[int(cls_id)] = {
                "class_id": int(cls_id),
                "class_name": cls_name,
                "precision": None,
                "recall": None,
                "f1": None,
                "AP50": None,
                "AP50_95": None,
            }

        for local_i, cls_id in enumerate(ap_class_index):
            p = safe_float(p_arr[local_i]) if local_i < len(p_arr) else None
            r = safe_float(r_arr[local_i]) if local_i < len(r_arr) else None
            f1 = safe_float(f1_arr[local_i]) if local_i < len(f1_arr) else None

            ap50 = None
            ap = None
            try:
                _, _, ap50_v, ap_v = box.class_result(local_i)
                ap50 = safe_float(ap50_v)
                ap = safe_float(ap_v)
            except Exception:
                pass

            if int(cls_id) not in base:
                base[int(cls_id)] = {
                    "class_id": int(cls_id),
                    "class_name": names_map.get(int(cls_id), f"class_{cls_id}"),
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "AP50": None,
                    "AP50_95": None,
                }

            base[int(cls_id)]["precision"] = p
            base[int(cls_id)]["recall"] = r
            base[int(cls_id)]["f1"] = f1
            base[int(cls_id)]["AP50"] = ap50
            base[int(cls_id)]["AP50_95"] = ap

        rows = [base[k] for k in sorted(base.keys())]

    except Exception:
        try:
            rows = results.summary()
        except Exception:
            rows = []

    if rows:
        pd.DataFrame(rows).to_csv(save_dir / "per_class_metrics.csv", index=False, encoding="utf-8-sig")


def export_per_class_generic(results, save_dir: Path) -> None:
    """
    seg 先通用导出 summary()。
    """
    try:
        rows = results.summary()
        if rows:
            pd.DataFrame(rows).to_csv(save_dir / "per_class_metrics.csv", index=False, encoding="utf-8-sig")
    except Exception:
        pass


def build_common_row(
    run_name: str,
    task: str,
    weight_path: Path,
    ckpt: dict,
    info: dict,
    results,
    imgsz: int,
) -> dict[str, Any]:
    speed = getattr(results, "speed", {}) or {}

    preprocess_ms = safe_float(speed.get("preprocess", None))
    inference_ms = safe_float(speed.get("inference", None))
    postprocess_ms = safe_float(speed.get("postprocess", None))
    loss_ms = safe_float(speed.get("loss", None))

    end2end_ms = None
    if preprocess_ms is not None and inference_ms is not None and postprocess_ms is not None:
        end2end_ms = preprocess_ms + inference_ms + postprocess_ms

    fps_inference = None
    if inference_ms is not None and inference_ms > 0:
        fps_inference = 1000.0 / inference_ms

    fps_end2end = None
    if end2end_ms is not None and end2end_ms > 0:
        fps_end2end = 1000.0 / end2end_ms

    row = {
        "run_name": run_name,
        "task": task,
        "weight_path": str(weight_path),
        "best_epoch": get_best_epoch_from_ckpt(ckpt),
        "imgsz_eval": imgsz,
        "params": info.get("params"),
        "flops_g": info.get("flops_g"),
        "weight_size_mb": round(get_weight_size_mb(weight_path), 3),
        "speed_preprocess_ms_per_img": preprocess_ms,
        "speed_inference_ms_per_img": inference_ms,
        "speed_postprocess_ms_per_img": postprocess_ms,
        "speed_loss_ms_per_img": loss_ms,
        "speed_end2end_ms_per_img": end2end_ms,
        "fps_inference_only": fps_inference,
        "fps_end2end": fps_end2end,
    }

    train_args = ckpt.get("train_args", {}) if isinstance(ckpt, dict) else {}
    for k in ["model", "data", "imgsz", "batch", "epochs", "device"]:
        if k in train_args:
            row[f"train_arg_{k}"] = train_args[k]

    return row


def build_det_row(common_row: dict[str, Any], results) -> dict[str, Any]:
    row = dict(common_row)
    row.update({
        "precision": safe_float(getattr(results.box, "mp", None)),
        "recall": safe_float(getattr(results.box, "mr", None)),
        "mAP50": safe_float(getattr(results.box, "map50", None)),
        "mAP50_95": safe_float(getattr(results.box, "map", None)),
    })
    return row


def build_seg_row(common_row: dict[str, Any], results) -> dict[str, Any]:
    row = dict(common_row)
    row.update({
        "box_precision": safe_float(getattr(results.box, "mp", None)),
        "box_recall": safe_float(getattr(results.box, "mr", None)),
        "box_mAP50": safe_float(getattr(results.box, "map50", None)),
        "box_mAP50_95": safe_float(getattr(results.box, "map", None)),
        "mask_precision": safe_float(getattr(results.seg, "mp", None)),
        "mask_recall": safe_float(getattr(results.seg, "mr", None)),
        "mask_mAP50": safe_float(getattr(results.seg, "map50", None)),
        "mask_mAP50_95": safe_float(getattr(results.seg, "map", None)),
    })
    return row


def save_metrics_json(row: dict[str, Any], save_dir: Path) -> None:
    with open(save_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False, indent=2)


# =============================================================================
# 3) 主流程
# =============================================================================

def main() -> None:
    ensure_dirs()

    print("当前 yolo26 仓库：", REPO_ROOT)
    print("训练结果根目录：", RUNS_ROOT)

    check_image_label_pairs(DET_VAL_IMAGES)
    check_image_label_pairs(SEG_VAL_IMAGES)

    det_yaml = DATA_YAML_DIR / "det_val.yaml"
    seg_yaml = DATA_YAML_DIR / "seg_val.yaml"
    build_val_yaml(DET_VAL_IMAGES, DET_NAMES, det_yaml)
    build_val_yaml(SEG_VAL_IMAGES, SEG_NAMES, seg_yaml)

    det_rows: list[dict[str, Any]] = []
    seg_rows: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    run_dirs = sorted([p for p in RUNS_ROOT.iterdir() if is_yolo26_run(p)])

    if not run_dirs:
        raise RuntimeError("没有找到任何 yolo26 训练结果目录。请检查 RUNS_ROOT 或 RUN_NAME_WHITELIST。")

    print("\n将评估以下 yolo26 目录：")
    for p in run_dirs:
        print(" -", p.name)

    for run_dir in run_dirs:
        run_name = run_dir.name
        print(f"\n{'=' * 100}")
        print(f"[开始] {run_name}")

        try:
            task = infer_task(run_dir)
            weight_path = find_best_weight(run_dir)
            if weight_path is None:
                raise FileNotFoundError(f"缺少 best.pt：{run_dir / 'weights' / 'best.pt'}")

            data_yaml = det_yaml if task == "detect" else seg_yaml
            imgsz = DET_IMGSZ if task == "detect" else SEG_IMGSZ
            fallback_names = DET_NAMES if task == "detect" else SEG_NAMES

            ckpt = load_ckpt_meta(weight_path)
            info = get_model_info(weight_path, imgsz)

            results, save_dir = run_val(
                weight_path=weight_path,
                data_yaml=data_yaml,
                imgsz=imgsz,
                run_name=run_name,
            )

            export_confusion_matrix(results, save_dir, fallback_names)

            if task == "detect":
                export_per_class_detect(results, save_dir, fallback_names)
            else:
                export_per_class_generic(results, save_dir)

            common_row = build_common_row(
                run_name=run_name,
                task=task,
                weight_path=weight_path,
                ckpt=ckpt,
                info=info,
                results=results,
                imgsz=imgsz,
            )

            if task == "detect":
                row = build_det_row(common_row, results)
                det_rows.append(row)
            else:
                row = build_seg_row(common_row, results)
                seg_rows.append(row)

            save_metrics_json(row, save_dir)

            print(f"[完成] {run_name}")
            print(f"输出目录：{save_dir}")

        except Exception as e:
            failed.append({
                "run_name": run_name,
                "error": str(e),
                "traceback": traceback.format_exc(),
            })
            print(f"[失败] {run_name}: {e}")

    if det_rows:
        det_df = pd.DataFrame(det_rows).sort_values("run_name")
        det_df.to_csv(EXPORT_ROOT / "summary_det_yolo26.csv", index=False, encoding="utf-8-sig")

    if seg_rows:
        seg_df = pd.DataFrame(seg_rows).sort_values("run_name")
        seg_df.to_csv(EXPORT_ROOT / "summary_seg_yolo26.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(EXPORT_ROOT / "summary_yolo26.xlsx", engine="openpyxl") as writer:
        if det_rows:
            pd.DataFrame(det_rows).sort_values("run_name").to_excel(
                writer, sheet_name="det_yolo26", index=False
            )
        if seg_rows:
            pd.DataFrame(seg_rows).sort_values("run_name").to_excel(
                writer, sheet_name="seg_yolo26", index=False
            )
        if failed:
            pd.DataFrame(failed).to_excel(writer, sheet_name="failed", index=False)

    if failed:
        with open(EXPORT_ROOT / "failed_runs.json", "w", encoding="utf-8") as f:
            json.dump(failed, f, ensure_ascii=False, indent=2)

    print("\n全部结束。")
    print("导出目录：", EXPORT_ROOT)


if __name__ == "__main__":
    main()