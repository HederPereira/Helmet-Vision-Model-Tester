"""
Wrapper para Faster R-CNN (ResNet50-FPN backbone) via torchvision.

Note a semelhança estrutural com SsdWrapper: como ambos usam a mesma API
do torchvision.models.detection, o essencial que muda entre os dois é
o build() (arquitetura) -- treino, avaliação e predict seguem o mesmo padrão.
"""

import time
from pathlib import Path
from typing import List, Union

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from core.base_model import BaseDetectorWrapper
from core.schemas import TrainConfig, TrainResult, EvalMetrics, Detection
from core.torchvision_common import build_dataloader, train_one_epoch, evaluate_map, build_optimizer


class FasterRcnnWrapper(BaseDetectorWrapper):

    model_name = "fasterrcnn_resnet50_fpn"

    def build(self, config: TrainConfig) -> None:
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

        n_classes_with_bg = self.num_classes + 1  # +1 = background

        model = fasterrcnn_resnet50_fpn(weights="DEFAULT" if config.pretrained else None)

        # Troca só a cabeça do ROI head para o nosso número de classes
        # (backbone + FPN + RPN pré-treinados são mantidos).
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes_with_bg)

        self.model = model.to(config.device)
        self._device = config.device

    # ------------------------------------------------------------------ #
    def train(self, dataset_dir: Union[str, Path], config: TrainConfig) -> TrainResult:
        if self.model is None:
            self.build(config)

        train_loader = build_dataloader(dataset_dir, "train", config, shuffle=True)
        val_loader = build_dataloader(dataset_dir, "val", config, shuffle=False)
        optimizer = build_optimizer(self.model, config)

        run_dir = Path(config.output_dir) / config.experiment_name
        run_dir.mkdir(parents=True, exist_ok=True)

        history = []
        best_map50 = -1.0
        best_ckpt_path = run_dir / "best.pt"
        epochs_without_improve = 0

        t0 = time.time()
        for epoch in range(1, config.epochs + 1):
            train_loss = train_one_epoch(self.model, optimizer, train_loader, config.device)

            map_result, inference_ms, fps = evaluate_map(
                self.model, val_loader, config.device, self.num_classes
            )
            map50 = float(map_result["map_50"])
            map50_95 = float(map_result["map"])

            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_map50": map50,
                "val_map50_95": map50_95,
            })

            if map50 > best_map50:
                best_map50 = map50
                torch.save(self.model.state_dict(), best_ckpt_path)
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1

            if epochs_without_improve >= config.patience:
                break  # early stopping

        train_time = time.time() - t0

        result = TrainResult(
            model_name=self.model_name,
            dataset_dir=str(dataset_dir),
            config=config.to_dict(),
            epochs_trained=len(history),
            train_time_s=train_time,
            checkpoint_path=str(best_ckpt_path),
            history=history,
            best_metrics={"map50": best_map50},
        )
        result.save()
        return result

    # ------------------------------------------------------------------ #
    def evaluate(self, dataset_dir: Union[str, Path], split: str = "test") -> EvalMetrics:
        if self.model is None:
            raise RuntimeError(f"[{self.model_name}] carregue um checkpoint antes de avaliar.")

        config = TrainConfig(device=self._device, batch_size=4)  # Faster R-CNN pesa mais em memória
        loader = build_dataloader(dataset_dir, split, config, shuffle=False)

        map_result, inference_ms, fps = evaluate_map(self.model, loader, config.device, self.num_classes)

        return EvalMetrics(
            model_name=self.model_name,
            dataset_dir=str(dataset_dir),
            split=split,
            map50=float(map_result["map_50"]),
            map50_95=float(map_result["map"]),
            precision=float(map_result.get("map_50", 0)),  # TODO: métrica dedicada (Fase 2)
            recall=float(map_result.get("mar_100", 0)),
            f1=0.0,  # TODO: calcular quando precision/recall dedicados existirem
            fps=fps,
            inference_time_ms=inference_ms,
            num_params=self.count_parameters(),
        )

    # ------------------------------------------------------------------ #
    def predict(
        self,
        source: Union[str, Path],
        conf_threshold: float = 0.25,
        save_output: bool = True,
        output_path: Union[str, Path, None] = None,
    ) -> List[Detection]:
        if self.model is None:
            raise RuntimeError(f"[{self.model_name}] carregue um checkpoint antes de prever.")

        self.model.eval()
        img = Image.open(source).convert("RGB")
        img_tensor = TF.to_tensor(img).to(self._device)

        with torch.no_grad():
            pred = self.model([img_tensor])[0]

        detections = []
        for box, score, label in zip(pred["boxes"], pred["scores"], pred["labels"]):
            if float(score) < conf_threshold:
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            cls_id = int(label) - 1
            detections.append(Detection(
                bbox=(x1, y1, x2, y2),
                score=float(score),
                class_id=cls_id,
                class_name=self.class_names[cls_id] if 0 <= cls_id < len(self.class_names) else str(cls_id),
            ))

        if save_output:
            self._save_annotated(img, detections, output_path or Path("results/_predictions") / Path(source).name)

        return detections

    def _save_annotated(self, img: Image.Image, detections: List[Detection], output_path):
        from PIL import ImageDraw
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        drawn = img.copy()
        draw = ImageDraw.Draw(drawn)
        for d in detections:
            draw.rectangle(d.bbox, outline="red", width=2)
            draw.text((d.bbox[0], max(0, d.bbox[1] - 10)), f"{d.class_name} {d.score:.2f}", fill="red")
        drawn.save(output_path)

    # ------------------------------------------------------------------ #
    def save_checkpoint(self, path: Union[str, Path]) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        return str(path)

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        if self.model is None:
            self.build(TrainConfig(pretrained=False))
        state_dict = torch.load(path, map_location=self._device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
