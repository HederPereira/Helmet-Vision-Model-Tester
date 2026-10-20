"""
Funções compartilhadas entre SsdWrapper e FasterRcnnWrapper.

Ambos os modelos do torchvision têm a MESMA interface de treino/inferência
(model(images, targets) -> dict de losses no modo train;
 model(images) -> list de predições no modo eval), então o loop de treino
e a avaliação de mAP podem ser 100% reaproveitados. Isso é o que garante
que SSD e Faster R-CNN sejam comparados sob exatamente as mesmas condições.
"""

import time
from typing import List

import torch
from torch.utils.data import DataLoader

from data.loaders.yolo_txt_dataset import YoloTxtDataset, collate_fn
from core.schemas import TrainConfig


def build_dataloader(dataset_dir: str, split: str, config: TrainConfig, shuffle: bool) -> DataLoader:
    dataset = YoloTxtDataset(dataset_dir, split=split)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
    )


def train_one_epoch(model, optimizer, data_loader: DataLoader, device: str) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)  # torchvision retorna dict de losses no modo train
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(1, n_batches)


@torch.no_grad()
def evaluate_map(model, data_loader: DataLoader, device: str, num_classes: int):
    """Calcula mAP@0.5 e mAP@0.5:0.95 usando torchmetrics, além de tempo
    médio de inferência por imagem (para FPS)."""
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)

    total_time = 0.0
    n_images = 0

    for images, targets in data_loader:
        images = [img.to(device) for img in images]

        t0 = time.time()
        preds = model(images)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        total_time += time.time() - t0
        n_images += len(images)

        preds_cpu = [{k: v.detach().cpu() for k, v in p.items()} for p in preds]
        targets_cpu = [{"boxes": t["boxes"], "labels": t["labels"]} for t in targets]
        metric.update(preds_cpu, targets_cpu)

    result = metric.compute()
    inference_time_ms = (total_time / max(1, n_images)) * 1000
    fps = 1000 / inference_time_ms if inference_time_ms > 0 else 0.0

    return result, inference_time_ms, fps


def build_optimizer(model, config: TrainConfig):
    params = [p for p in model.parameters() if p.requires_grad]
    if config.optimizer.lower() == "adam":
        return torch.optim.Adam(params, lr=config.learning_rate)
    if config.optimizer.lower() == "adamw":
        return torch.optim.AdamW(params, lr=config.learning_rate)
    return torch.optim.SGD(params, lr=config.learning_rate, momentum=0.9, weight_decay=5e-4)
