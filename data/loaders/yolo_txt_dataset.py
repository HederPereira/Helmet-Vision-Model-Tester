"""
Dataset comum para SSD e Faster R-CNN (torchvision).

Lê a MESMA estrutura de diretório definida em core/DATA_CONTRACT.md
(a mesma que o YoloWrapper consome via data.yaml), convertendo
as bboxes normalizadas (formato YOLO) para pixels absolutos e para o
dicionário de target que os modelos de detecção do torchvision esperam:
    target = {"boxes": FloatTensor[N,4], "labels": Int64Tensor[N]}

Isso é o que garante que os 3 modelos vejam EXATAMENTE as mesmas imagens
e as mesmas anotações no benchmark.
"""

from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class YoloTxtDataset(Dataset):

    IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

    def __init__(self, dataset_dir: str, split: str = "train", transforms=None):
        """
        split: "train" | "val" | "test"
        transforms: callable opcional (img, target) -> (img, target).
                    Se None, aplica só ToTensor (sem augmentation).
        """
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.transforms = transforms

        self.images_dir = self.dataset_dir / "images" / split
        self.labels_dir = self.dataset_dir / "labels" / split
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Diretório de imagens não encontrado: {self.images_dir}")

        self.image_paths: List[Path] = sorted(
            p for p in self.images_dir.iterdir() if p.suffix.lower() in self.IMG_EXTENSIONS
        )

        classes_file = self.dataset_dir / "classes.txt"
        self.class_names = [
            l.strip() for l in classes_file.read_text(encoding="utf-8").splitlines() if l.strip()
        ]
        # torchvision reserva a classe 0 para "background" nos modelos pré-treinados
        # padrão (Faster R-CNN / SSD do torchvision) -> deslocamos os ids em +1.
        self.background_offset = 1

    def __len__(self) -> int:
        return len(self.image_paths)

    def _label_path_for(self, img_path: Path) -> Path:
        return self.labels_dir / f"{img_path.stem}.txt"

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        boxes, labels = [], []
        label_path = self._label_path_for(img_path)
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                cls_id, xc, yc, bw, bh = line.split()
                cls_id = int(cls_id)
                xc, yc, bw, bh = map(float, (xc, yc, bw, bh))

                x1 = (xc - bw / 2) * w
                y1 = (yc - bh / 2) * h
                x2 = (xc + bw / 2) * w
                y2 = (yc + bh / 2) * h

                boxes.append([x1, y1, x2, y2])
                labels.append(cls_id + self.background_offset)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
            "orig_size": (w, h),
            "path": str(img_path),
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        else:
            img = TF.to_tensor(img)

        return img, target

    @property
    def num_classes_with_background(self) -> int:
        return len(self.class_names) + self.background_offset


def collate_fn(batch):
    """Necessário porque cada imagem pode ter um número diferente de boxes
    -- não dá para empilhar em um tensor único como no batch de classificação."""
    return tuple(zip(*batch))
