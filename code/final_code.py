import argparse, json, time
from pathlib import Path

import numpy as np
import skimage.io
import skimage.transform as skt

from selective_search import selective_search

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, average_precision_score
)
import joblib

import torch
import torchvision.models as models
from torch import nn
from PIL import Image
import numpy as _np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ---------------------------
# Shared helpers
# ---------------------------
def read_image(img_path):
    img = skimage.io.imread(str(img_path))
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    return img

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]

def iou_xywh(a, b):
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = xywh_to_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union == 0 else inter / union

def safe_crop(img, box):
    H, W = img.shape[:2]
    x, y, w, h = [int(round(v)) for v in box]
    x = max(0, min(W - 1, x))
    y = max(0, min(H - 1, y))
    w = max(1, min(W - x, w))
    h = max(1, min(H - y, h))
    return img[y:y+h, x:x+w]

def load_coco(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    images = {im["id"]: im for im in coco.get("images", [])}
    anns = coco.get("annotations", [])
    gts = {}
    for a in anns:
        gts.setdefault(a["image_id"], []).append(a["bbox"])
    return images, gts


# ---------------------------
# Task 5.2.1 — proposals
# ---------------------------
def run_ss(img_path, scale, sigma, min_size, max_props):
    image = read_image(img_path)
    _, regions = selective_search(image, scale=scale, sigma=sigma, min_size=min_size)

    seen, boxes = set(), []
    for r in regions:
        x, y, w, h = r["rect"]
        if w < 5 or h < 5:
            continue
        tup = (int(x), int(y), int(w), int(h))
        if tup in seen:
            continue
        seen.add(tup)
        boxes.append(list(tup))
        if len(boxes) >= max_props:
            break
    return boxes

def process_split_proposals(data_root, split, out_dir, scale, sigma, min_size, max_props):
    ann = Path(data_root) / split / "_annotations.coco.json"
    if not ann.exists():
        raise FileNotFoundError(f"Missing file: {ann}")

    images, _ = load_coco(ann)
    results, failures = [], []
    start = time.time()

    for i, (img_id, meta) in enumerate(images.items(), 1):
        candidate = Path(data_root) / split / meta["file_name"]
        if not candidate.exists():
            candidate = Path(data_root) / meta["file_name"]
        if not candidate.exists():
            failures.append({"image_id": img_id, "file_name": meta["file_name"], "error": "file not found"})
            continue
        try:
            props = run_ss(candidate, scale, sigma, min_size, max_props)
            results.append(
                {"image_id": img_id, "file_name": meta["file_name"],
                 "width": meta.get("width"), "height": meta.get("height"),
                 "proposals": props}
            )
        except Exception as e:
            failures.append({"image_id": img_id, "file_name": meta["file_name"], "error": f"{type(e).__name__}: {e}"})
        if i % 5 == 0 or i == len(images):
            print(f"[{split}] processed {i}/{len(images)} images")

    took = time.time() - start
    avg_props = (sum(len(r["proposals"]) for r in results) / max(1, len(results)))
    print(f"[{split}] done in {took:.1f}s | images: {len(results)} | avg props: {avg_props:.1f} | failures: {len(failures)}")

    ensure_dir(out_dir)
    out_path = Path(out_dir) / f"proposals_{split}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"split": split, "dataset_root": str(data_root),
                   "params": {"scale": scale, "sigma": sigma, "min_size": min_size, "max_props": max_props},
                   "results": results, "failures": failures}, f)
    print(f"saved: {out_path}")
    return out_path


# ---------------------------
# Task 5.2.2 — samples via IoU
# ---------------------------
def build_samples_for_split(data_root, split, proposals_json, samples_dir, tp=0.75, tn=0.25,
                            save_crops=True, crop_size=128, max_neg_per_image=None):
    data_root = Path(data_root)
    samples_dir = Path(samples_dir) / split
    pos_dir = samples_dir / "pos"
    neg_dir = samples_dir / "neg"
    if save_crops:
        ensure_dir(pos_dir); ensure_dir(neg_dir)

    with open(proposals_json, "r", encoding="utf-8") as f:
        prop = json.load(f)
    assert prop["split"] == split, f"Split mismatch: {proposals_json}"

    ann = Path(data_root) / split / "_annotations.coco.json"
    images, gts = load_coco(ann)

    meta, pos_count, neg_count = [], 0, 0

    for item in prop["results"]: #loop over the images listed in proposal
        img_id = item["image_id"]
        file_name = item["file_name"]
        gt_boxes = gts.get(img_id, [])
        img_path = (data_root / split / file_name)
        if not img_path.exists(): img_path = data_root / file_name
        img = read_image(img_path)

        neg_kept = 0
        for idx, box in enumerate(item["proposals"]):
            max_iou = max([iou_xywh(box, gt) for gt in gt_boxes], default=0.0)
            label = 1 if max_iou >= tp else 0 if max_iou <= tn else None
            if label is None: continue
            if label == 0 and max_neg_per_image is not None and neg_kept >= max_neg_per_image: continue

            entry = {"image_id": img_id, "file_name": file_name,
                     "bbox": [int(x) for x in box], "iou": float(max_iou),
                     "label": int(label), "crop_path": None}

            if save_crops:
                crop = safe_crop(img, box)
                if crop_size is not None:
                    crop = skt.resize(crop, (crop_size, crop_size), anti_aliasing=True, preserve_range=True).astype(np.uint8)
                outp = (pos_dir if label == 1 else neg_dir) / f"{img_id}_{idx}_{int(round(max_iou*100))}.jpg"
                skimage.io.imsave(outp.as_posix(), crop, check_contrast=False)
                entry["crop_path"] = str(outp)

            meta.append(entry)
            pos_count += (label == 1); neg_count += (label == 0); neg_kept += (label == 0)

    ensure_dir(samples_dir)
    meta_path = samples_dir / f"samples_{split}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"split": split, "tp": tp, "tn": tn, "save_crops": save_crops,
                   "crop_size": crop_size, "data_root": str(data_root), "samples": meta}, f)
    print(f"[{split}] positives: {pos_count} | negatives: {neg_count} | saved meta: {meta_path}")
    return meta_path


# ---------------------------
# Task 5.2.2 — ResNet18 feature extraction
# ---------------------------
def _build_resnet18(device="cpu"):
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    m.fc = nn.Identity()     # 512-d embedding
    dim = 512
    size = 224
    normalize = models.ResNet18_Weights.DEFAULT.transforms()
    m.eval().to(device)
    return m, dim, size, normalize

def _load_or_crop(sample, data_root, split, target_size=224):
    if sample.get("crop_path") and Path(sample["crop_path"]).exists():
        arr = read_image(sample["crop_path"])
    else:
        img_path = Path(data_root) / split / sample["file_name"]
        if not img_path.exists():
            img_path = Path(data_root) / sample["file_name"]
        img = read_image(img_path)
        arr = safe_crop(img, sample["bbox"])
    if target_size is not None:
        arr = skt.resize(arr, (target_size, target_size), anti_aliasing=True, preserve_range=True).astype(np.uint8)
    return arr

def to_pil(arr): return Image.fromarray(arr)

def extract_features_resnet18(samples_json, out_dir, data_root, split, batch_size=64, device="cpu"):
    with open(samples_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    samples = meta["samples"]

    model, dim, size, normalize = _build_resnet18(device)
    feats, labels = [], []

    batch_imgs, batch_idx = [], []
    for i, s in enumerate(samples, 1):
        arr = _load_or_crop(s, data_root, split, target_size=size)
        pil = to_pil(arr)
        tensor = normalize(pil).unsqueeze(0)  # (1,3,H,W)
        batch_imgs.append(tensor); batch_idx.append(i)

        if len(batch_imgs) == batch_size or i == len(samples):
            x = torch.cat(batch_imgs, dim=0).to(device)
            with torch.no_grad():
                z = model(x).detach().cpu().numpy()  # (B, dim)
            feats.append(z)
            labels.extend([int(samples[j-1]["label"]) for j in batch_idx])
            batch_imgs, batch_idx = [], []

        if i % 200 == 0 or i == len(samples):
            print(f"[{split}] CNN processed {i}/{len(samples)}")

    X = np.concatenate(feats, axis=0).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    ensure_dir(out_dir)
    out_path = Path(out_dir) / f"features_{split}_resnet18.npz"
    np.savez_compressed(out_path, X=X, y=y, extractor="resnet18", dim=int(dim))
    print(f"[{split}] resnet18 features: {X.shape} -> {out_path}")
    return out_path

def _load_features(features_dir, split, extractor="resnet18"):
    p = Path(features_dir) / f"features_{split}_{extractor}.npz"
    if not p.exists():
        raise FileNotFoundError(f"Feature file not found: {p}")
    data = np.load(p)
    return data["X"], data["y"]


# ---------------------------
# Training SVM
# ---------------------------
def train_svm(features_dir, extractor="resnet18", model_out="models/svm.joblib",
              valid_split="valid", kernel="linear", C=1.0, probability=False,
              use_pca=False, pca_components=128, random_state=42):

    X_tr, y_tr = _load_features(features_dir, "train", extractor)
    X_va, y_va = _load_features(features_dir, valid_split, extractor)

    steps = [("scaler", StandardScaler())]
    if use_pca:
        steps.append(("pca", PCA(n_components=pca_components, random_state=random_state)))

    if kernel == "linear" and not probability:
        clf = LinearSVC(C=C, class_weight="balanced", random_state=random_state, dual=False)
    #else:
    #    clf = SVC(C=C, kernel=kernel, class_weight="balanced",
    #              probability=probability, random_state=random_state)
    steps.append(("svm", clf))
    pipe = Pipeline(steps)

    print(f"Training SVM on {X_tr.shape} (extractor=resnet18, kernel={kernel}, C={C}, "
          f"pca={'on' if use_pca else 'off'})")
    pipe.fit(X_tr, y_tr)

    y_pred = pipe.predict(X_va)
    acc = accuracy_score(y_va, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_va, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_va, y_pred).tolist()

    ap = None
    try:
        if probability:
            scores = pipe.predict_proba(X_va)[:, 1]
        else:
            scores = pipe.decision_function(X_va)
        ap = float(average_precision_score(y_va, scores))
    except Exception:
        pass

    print("\nValidation report:")
    print(classification_report(y_va, y_pred, digits=4))
    print("Confusion matrix:", cm)
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | AP: {ap}")

    model_out = Path(model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_out)
    metrics_path = model_out.with_suffix(".metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "extractor": "resnet18",
            "kernel": kernel,
            "C": C,
            "probability": probability,
            "use_pca": use_pca,
            "pca_components": pca_components,
            "train_shape": [int(x) for x in X_tr.shape],
            "valid_shape": [int(x) for x in X_va.shape],
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "AP": ap,
            "confusion_matrix": cm
        }, f, indent=2)
    print(f"\nSaved model: {model_out}\nSaved metrics: {metrics_path}")
    return str(model_out), str(metrics_path)


# ---------------------------
# Inference 
# ---------------------------
def _model_has_proba(pipe):
    try:
        return hasattr(pipe, "predict_proba") or hasattr(pipe[-1], "predict_proba")
    except Exception:
        return False

def _pipe_predict_scores(pipe, X, use_sigmoid=True):
    if _model_has_proba(pipe):
        try:
            proba = pipe.predict_proba(X)[:, 1]
            return proba.astype(_np.float32)
        except Exception:
            pass
    if hasattr(pipe, "decision_function"):
        dec = pipe.decision_function(X).astype(_np.float32)
    else:
        dec = pipe[:].decision_function(X).astype(_np.float32)
    if use_sigmoid:
        return (1.0 / (1.0 + _np.exp(-dec))).astype(_np.float32)
    return dec

def non_max_suppression(boxes, scores, iou_thresh=0.3):
    if len(boxes) == 0:
        return []
    boxes = _np.asarray(boxes, dtype=_np.float32)
    scores = _np.asarray(scores, dtype=_np.float32)
    x1 = boxes[:,0]; y1 = boxes[:,1]
    x2 = boxes[:,0] + boxes[:,2]; y2 = boxes[:,1] + boxes[:,3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = _np.maximum(x1[i], x1[order[1:]])
        yy1 = _np.maximum(y1[i], y1[order[1:]])
        xx2 = _np.minimum(x2[i], x2[order[1:]])
        yy2 = _np.minimum(y2[i], y2[order[1:]])
        w = _np.maximum(0.0, xx2 - xx1)
        h = _np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = _np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]
    return keep

def _resnet18_feature_batch(crops, device="cpu"):
    model, dim, size, normalize = _build_resnet18(device)
    tensors = []
    for arr in crops:
        pil = Image.fromarray(arr)
        tensors.append(normalize(pil).unsqueeze(0))
    x = torch.cat(tensors, dim=0).to(device)
    with torch.no_grad():
        z = model(x).detach().cpu().numpy().astype(np.float32)
    return z, size

def _clip_box_to_image(box, W, H):
    x,y,w,h = box
    x = max(0, min(W-1, int(round(x))))
    y = max(0, min(H-1, int(round(y))))
    w = max(1, min(W-x, int(round(w))))
    h = max(1, min(H-y, int(round(h))))
    return [x,y,w,h]

def infer_on_image(
    image_path,
    model_path,
    score_thr=0.5,
    nms_iou=0.3,
    top_k=200,
    scale=200, sigma=0.9, min_size=50, max_props=1500,
    device="cpu",
):
    img = read_image(image_path)
    H, W = img.shape[:2]
    _, regions = selective_search(img, scale=scale, sigma=sigma, min_size=min_size)
    props, seen = [], set()
    for r in regions:
        x,y,w,h = r["rect"]
        if w < 5 or h < 5: continue
        tup = _clip_box_to_image([x,y,w,h], W, H)
        key = tuple(tup)
        if key in seen: continue
        seen.add(key); props.append(tup)
        if len(props) >= max_props: break
    if not props:
        return {"image": image_path, "detections": []}, img

    crops = []
    for b in props:
        crop = safe_crop(img, b)
        crops.append(crop)

    feats, size = _resnet18_feature_batch(crops, device=device)

    pipe = joblib.load(model_path)
    scores = _pipe_predict_scores(pipe, feats, use_sigmoid=True)

    order = np.argsort(scores)[::-1][:min(top_k, len(scores))]
    props_top = [props[i] for i in order]
    scores_top = [float(scores[i]) for i in order]
    filt = [(b,s) for b,s in zip(props_top, scores_top) if s >= score_thr]
    if not filt:
        return {"image": image_path, "detections": []}, img
    boxes_f, scores_f = zip(*filt)
    keep_idx = non_max_suppression(list(boxes_f), list(scores_f), iou_thresh=nms_iou)
    final = [{"bbox": [int(x) for x in boxes_f[i]], "score": float(scores_f[i])} for i in keep_idx]

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img)
    for det in final:
        x,y,w,h = det["bbox"]
        ax.add_patch(Rectangle((x,y), w, h, fill=False, linewidth=2))
        ax.text(x, y-3, f"{det['score']:.2f}", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))
    ax.axis("off")
    fig.canvas.draw()
    vis = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    vis = vis[:, :, :3].copy()
    plt.close(fig)

    return {
        "image": image_path,
        "detections": final,
        "num_proposals": len(props),
        "kept_after_thr": len(filt),
        "kept_after_nms": len(final),
    }, vis


# ---------------------------
# CLI
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Selective Search + ResNet18 features + SVM (train/infer).")
    ap.add_argument("--mode", choices=["proposals", "samples", "features", "train", "infer"], default="proposals")
    ap.add_argument("--data-root", required=False, help="Required for: proposals, samples, features")
    ap.add_argument("--splits", nargs="+", default=["train", "valid"])

    # proposals
    ap.add_argument("--out-dir", default="proposals")
    ap.add_argument("--scale", type=float, default=200.0)
    ap.add_argument("--sigma", type=float, default=0.9)
    ap.add_argument("--min-size", type=int, default=50)
    ap.add_argument("--max-props", type=int, default=1500)

    # samples
    ap.add_argument("--proposals-dir", default="proposals")
    ap.add_argument("--samples-dir", default="samples")
    ap.add_argument("--tp", type=float, default=0.75)
    ap.add_argument("--tn", type=float, default=0.25)
    ap.add_argument("--save-crops", action="store_true")
    ap.add_argument("--crop-size", type=int, default=128)
    ap.add_argument("--max-neg-per-image", type=int, default=None)

    # features
    ap.add_argument("--features-dir", default="features")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cpu")  # or "cuda"

    # training
    ap.add_argument("--model-out", default="models/svm.joblib")
    ap.add_argument("--kernel", choices=["linear", "rbf"], default="linear")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--probability", action="store_true")
    ap.add_argument("--use-pca", action="store_true")
    ap.add_argument("--pca-components", type=int, default=128)

    # inference
    ap.add_argument("--model", help="Path to trained SVM pipeline (.joblib) for inference")
    ap.add_argument("--images", nargs="+", help="Image files and/or folders to run inference on")
    ap.add_argument("--score-thr", type=float, default=0.5)
    ap.add_argument("--nms-iou", type=float, default=0.3)
    ap.add_argument("--top-k", type=int, default=200)

    args = ap.parse_args()

    if args.mode == "proposals":
        if not args.data_root:
            ap.error("--data-root is required for mode 'proposals'")
        data_root = Path(args.data_root)
        for split in args.splits:
            process_split_proposals(
                data_root=data_root, split=split, out_dir=args.out_dir,
                scale=args.scale, sigma=args.sigma, min_size=args.min_size, max_props=args.max_props
            )

    elif args.mode == "samples":
        if not args.data_root:
            ap.error("--data-root is required for mode 'samples'")
        data_root = Path(args.data_root)
        for split in args.splits:
            proposals_json = Path(args.proposals_dir) / f"proposals_{split}.json"
            if not proposals_json.exists():
                raise FileNotFoundError(f"Missing proposals file: {proposals_json}")
            build_samples_for_split(
                data_root=data_root, split=split, proposals_json=proposals_json,
                samples_dir=args.samples_dir, tp=args.tp, tn=args.tn,
                save_crops=args.save_crops, crop_size=args.crop_size,
                max_neg_per_image=args.max_neg_per_image
            )

    elif args.mode == "features":
        if not args.data_root:
            ap.error("--data-root is required for mode 'features'")
        data_root = Path(args.data_root)
        feats_dir = Path(args.features_dir); ensure_dir(feats_dir)
        for split in args.splits:
            samples_json = Path(args.samples_dir) / split / f"samples_{split}.json"
            if not samples_json.exists():
                raise FileNotFoundError(f"Missing samples meta: {samples_json}")
            extract_features_resnet18(samples_json, feats_dir, data_root, split,
                                      batch_size=args.batch_size, device=args.device)

    elif args.mode == "train":
        train_svm(
            features_dir=args.features_dir,
            extractor="resnet18",
            model_out=args.model_out,
            valid_split="valid",
            kernel=args.kernel,
            C=args.C,
            probability=args.probability,
            use_pca=args.use_pca,
            pca_components=args.pca_components,
        )

    else:  # infer
        if not args.model:
            ap.error("--model is required for mode 'infer'")
        if not args.images:
            ap.error("--images is required for mode 'infer'")

        # Expand file list
        files = []
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        for inp in args.images:
            p = Path(inp).expanduser()
            if p.is_dir():
                files.extend([str(q) for q in p.rglob("*") if q.suffix.lower() in exts])
            elif p.exists():
                files.append(str(p))
        if not files:
            raise FileNotFoundError(f"No input images found for inference under: {args.images}")

        out_dir = Path(args.out_dir if hasattr(args, "out_dir") else "detections").expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)

        model_path = str(Path(args.model).expanduser())
        all_results = []
        for path in files:
            det, vis = infer_on_image(
                image_path=path,
                model_path=model_path,
                score_thr=args.score_thr,
                nms_iou=args.nms_iou,
                top_k=args.top_k,
                scale=args.scale, sigma=args.sigma, min_size=args.min_size, max_props=args.max_props,
                device=args.device,
            )
            all_results.append(det)
            fn = Path(path).stem + "_det.png"
            skimage.io.imsave(str(out_dir / fn), vis, check_contrast=False)
            print(f"[infer] Saved visualization → {out_dir / fn}")

        with open(out_dir / "detections.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"[infer] Wrote detections JSON → {out_dir / 'detections.json'}")


if __name__ == "__main__":
    main()
