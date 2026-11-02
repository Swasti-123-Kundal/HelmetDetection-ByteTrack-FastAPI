"""Convert a Pascal-VOC style dataset (images/ + annotations/*.xml) to YOLO format.

Works with the Kaggle dataset "Helmet Detection" (andrewmvd/helmet-detection):
    kaggle datasets download -d andrewmvd/helmet-detection -p raw --unzip
    python scripts/voc_to_yolo.py --src raw --out datasets/helmet

Class names are read from the XML files and normalised ("Without Helmet" -> without_helmet),
so you do not have to hard-code them. Check the printed class list and make sure
`violation_classes` in configs/project.yaml contains the "no helmet" one.
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from saferide.violations import normalize  # noqa: E402

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")


def find_image(images_dir: Path, stem: str, hint: str | None) -> Path | None:
    if hint and (images_dir / hint).is_file():
        return images_dir / hint
    for ext in IMG_EXT:
        p = images_dir / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def parse_xml(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w, h = int(float(size.findtext("width"))), int(float(size.findtext("height")))
    objs = []
    for o in root.findall("object"):
        bb = o.find("bndbox")
        objs.append(
            (
                normalize(o.findtext("name", "")),
                float(bb.findtext("xmin")), float(bb.findtext("ymin")),
                float(bb.findtext("xmax")), float(bb.findtext("ymax")),
            )
        )
    return w, h, objs, root.findtext("filename")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="folder containing images/ and annotations/")
    ap.add_argument("--out", default="datasets/helmet")
    ap.add_argument("--val", type=float, default=0.2)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    images_dir, ann_dir = src / "images", src / "annotations"
    if not images_dir.is_dir() or not ann_dir.is_dir():
        raise SystemExit(f"Expected {images_dir} and {ann_dir} to exist")

    samples = []
    for xml in sorted(ann_dir.glob("*.xml")):
        w, h, objs, fname = parse_xml(xml)
        img = find_image(images_dir, xml.stem, fname)
        if img is None or not objs or w <= 0 or h <= 0:
            continue
        samples.append((img, w, h, objs))
    if not samples:
        raise SystemExit("No usable image/annotation pairs found")

    classes = sorted({o[0] for s in samples for o in s[3]})
    cid = {c: i for i, c in enumerate(classes)}

    random.Random(args.seed).shuffle(samples)
    n = len(samples)
    n_test, n_val = int(n * args.test), int(n * args.val)
    splits = {
        "test": samples[:n_test],
        "val": samples[n_test:n_test + n_val],
        "train": samples[n_test + n_val:],
    }

    for split, items in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img, w, h, objs in items:
            shutil.copy(img, out / "images" / split / img.name)
            lines = []
            for name, x1, y1, x2, y2 in objs:
                x1, x2 = max(0.0, min(x1, w)), max(0.0, min(x2, w))
                y1, y2 = max(0.0, min(y1, h)), max(0.0, min(y2, h))
                bw, bh = x2 - x1, y2 - y1
                if bw < 1 or bh < 1:
                    continue
                lines.append(f"{cid[name]} {(x1 + bw / 2) / w:.6f} {(y1 + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}")
            (out / "labels" / split / f"{img.stem}.txt").write_text("\n".join(lines) + "\n")

    names_block = "\n".join(f"  {i}: {c}" for c, i in cid.items())
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names_block}\n"
    )
    print(f"classes: {classes}")
    print({k: len(v) for k, v in splits.items()}, f"-> {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
