#!/usr/bin/env python
"""Nível 2: valida a conversão YOLO-txt -> torchvision feita pelo YoloTxtDataset.

É o ponto de falha mais perigoso do benchmark: se a conversão estiver errada,
SSD e Faster R-CNN treinam com caixas erradas e a comparação com o YOLO fica
inválida **sem levantar erro nenhum**.

Além dos asserts, gera imagens com as caixas convertidas desenhadas de volta —
olhe para elas. Números passando em assert não provam que o eixo não está invertido.

Uso: .venv/bin/python scripts/check_contract.py [dataset_dir] [split]
"""

import _bootstrap  # noqa: F401

import sys
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import ImageDraw
from torch.utils.data import DataLoader

from data.loaders.yolo_txt_dataset import YoloTxtDataset, collate_fn

OUT_DIR = Path("results/_contract_check")


def main(dataset_dir: str, split: str) -> int:
    ds = YoloTxtDataset(dataset_dir, split=split)
    print(f"dataset   {dataset_dir} (split={split})")
    print(f"imagens   {len(ds)}")
    print(f"classes   {ds.class_names}  (+background = {ds.num_classes_with_background})")

    img, target = ds[0]
    C, H, W = img.shape
    boxes, labels = target["boxes"], target["labels"]
    print(f"img[0]    {tuple(img.shape)} {img.dtype} min={float(img.min()):.3f} max={float(img.max()):.3f}")
    print(f"boxes[0]  {tuple(boxes.shape)}  labels={labels.tolist()}")

    assert C == 3, "imagem deve ter 3 canais"
    assert img.dtype == torch.float32 and 0.0 <= float(img.min()) and float(img.max()) <= 1.0, \
        "imagem deve ser float32 normalizada em [0,1]"
    assert boxes.ndim == 2 and boxes.shape[1] == 4, "boxes deve ser [N,4]"
    assert (boxes[:, 2] > boxes[:, 0]).all() and (boxes[:, 3] > boxes[:, 1]).all(), \
        "formato xyxy: x2>x1 e y2>y1"
    assert boxes.min() >= -1 and boxes[:, 2].max() <= W + 1 and boxes[:, 3].max() <= H + 1, \
        "boxes em pixels absolutos, dentro da imagem"
    assert labels.min() >= 1, "labels devem vir deslocados em +1 (0 = background)"
    assert labels.max() <= len(ds.class_names), "label fora do range de classes"
    print("OK  boxes em pixels absolutos (xyxy), labels com offset de background")

    imgs, targets = collate_fn([ds[0], ds[1]])
    assert isinstance(imgs, tuple) and len(imgs) == 2
    print("OK  collate_fn preserva N variável de boxes por imagem")

    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)
    imgs, targets = next(iter(loader))
    print(f"OK  DataLoader entrega {len(imgs)} imagens por batch")

    # conferência visual
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(min(3, len(ds))):
        img, t = ds[i]
        pil = TF.to_pil_image(img)
        draw = ImageDraw.Draw(pil)
        for box, lab in zip(t["boxes"].tolist(), t["labels"].tolist()):
            draw.rectangle(box, outline="lime", width=3)
            draw.text((box[0], max(0, box[1] - 12)), ds.class_names[int(lab) - 1], fill="lime")
        pil.save(OUT_DIR / f"check_{i}.jpg")
    print(f"\n>> ABRA as imagens em {OUT_DIR}/ — as caixas devem cercar os objetos")
    print("   e o nome da classe deve bater com o objeto certo.")
    return 0


if __name__ == "__main__":
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else "datasets/dummy_helmet"
    split = sys.argv[2] if len(sys.argv) > 2 else "train"
    sys.exit(main(dataset_dir, split))
