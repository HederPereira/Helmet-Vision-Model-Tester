"""Nível 1: gera um dataset sintético mínimo no formato do core/DATA_CONTRACT.md.

Imagens com elipses coloridas e labels YOLO válidos — serve para exercitar o
caminho inteiro do código em segundos, sem esperar pelo dataset real.
Um modelo treinado aqui terá mAP baixo, e isso é esperado.

Uso: .venv/bin/python scripts/make_dummy_dataset.py [destino] [n_train] [n_val] [n_test]
"""

import _bootstrap  # noqa: F401

import random
import sys
from pathlib import Path
from PIL import Image, ImageDraw

CLASSES = ["com_capacete", "sem_capacete"]
COLORS = [(230, 60, 60), (60, 120, 230)]


def make_split(root: Path, split: str, n: int, rng: random.Random, size=(320, 320)):
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n):
        W, H = size
        img = Image.new("RGB", (W, H), (245, 245, 235))
        draw = ImageDraw.Draw(img)
        lines = []
        for _ in range(rng.randint(1, 3)):
            cls = rng.randrange(len(CLASSES))
            bw, bh = rng.randint(40, 110), rng.randint(40, 110)
            x1 = rng.randint(0, W - bw)
            y1 = rng.randint(0, H - bh)
            x2, y2 = x1 + bw, y1 + bh
            draw.ellipse([x1, y1, x2, y2], fill=COLORS[cls])
            lines.append(
                f"{cls} {((x1+x2)/2)/W:.6f} {((y1+y2)/2)/H:.6f} {bw/W:.6f} {bh/H:.6f}"
            )
        name = f"{split}_{i:04d}"
        img.save(img_dir / f"{name}.jpg", quality=90)
        (lbl_dir / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/dummy_helmet")
    n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    n_val = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    n_test = int(sys.argv[4]) if len(sys.argv) > 4 else 4

    rng = random.Random(42)
    for split, n in (("train", n_train), ("val", n_val), ("test", n_test)):
        make_split(root, split, n, rng)
    (root / "classes.txt").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")
    print(f"Dataset sintético criado em: {root.resolve()}")
    print(f"  train={n_train}  val={n_val}  test={n_test}  classes={CLASSES}")


if __name__ == "__main__":
    main()
