import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

XML = """<annotation><filename>{n}.png</filename><size><width>200</width><height>100</height></size>
<object><name>With Helmet</name><bndbox><xmin>10</xmin><ymin>20</ymin><xmax>60</xmax><ymax>70</ymax></bndbox></object>
<object><name>Without Helmet</name><bndbox><xmin>100</xmin><ymin>0</ymin><xmax>150</xmax><ymax>50</ymax></bndbox></object>
</annotation>"""


def test_conversion(tmp_path):
    src = tmp_path / "raw"
    (src / "images").mkdir(parents=True)
    (src / "annotations").mkdir()
    for i in range(10):
        cv2.imwrite(str(src / "images" / f"BikesHelmets{i}.png"), np.zeros((100, 200, 3), np.uint8))
        (src / "annotations" / f"BikesHelmets{i}.xml").write_text(XML.format(n=f"BikesHelmets{i}"))
    out = tmp_path / "ds"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "voc_to_yolo.py"), "--src", str(src), "--out", str(out)],
        check=True,
    )
    yaml_text = (out / "data.yaml").read_text()
    assert "without_helmet" in yaml_text and "with_helmet" in yaml_text
    label = next((out / "labels" / "train").glob("*.txt")).read_text().strip().splitlines()
    assert len(label) == 2
    cls, xc, yc, w, h = label[0].split()
    # first object: with_helmet (class 0 after sorting), box 10..60 x 20..70 in a 200x100 image
    assert cls == "0"
    assert abs(float(xc) - 35 / 200) < 1e-5 and abs(float(yc) - 45 / 100) < 1e-5
    assert abs(float(w) - 50 / 200) < 1e-5 and abs(float(h) - 50 / 100) < 1e-5
    n_img = sum(len(list((out / "images" / s).glob("*.png"))) for s in ("train", "val", "test"))
    assert n_img == 10
