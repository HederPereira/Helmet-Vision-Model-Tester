"""
Estruturas de dados compartilhadas entre todos os wrappers de modelo.

Manter essas classes desacopladas de qualquer framework (torch, ultralytics, etc.)
é o que permite que o restante do sistema (trainer, evaluator, UI) trate
YOLO, SSD e Faster R-CNN de forma intercambiável.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import json
import time


@dataclass
class TrainConfig:
    """Configuração de treino. Nem todo campo é usado por todo modelo
    (ex: YOLO ignora `optimizer` se usar o default do Ultralytics),
    mas manter os campos aqui garante que o experimento seja documentado
    de forma consistente entre arquiteturas."""

    epochs: int = 50
    batch_size: int = 16
    img_size: int = 640
    learning_rate: float = 1e-3
    optimizer: str = "SGD"          # SGD | Adam | AdamW
    device: str = "cuda:0"          # ou "cpu"
    pretrained: bool = True
    num_workers: int = 4
    patience: int = 10              # early stopping (quando suportado)
    seed: int = 42
    output_dir: str = "results"     # onde salvar checkpoints/logs deste run
    experiment_name: str = "exp"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Detection:
    """Uma única detecção retornada por predict()."""

    bbox: tuple  # (x1, y1, x2, y2) em pixels, coordenadas da imagem original
    score: float
    class_id: int
    class_name: str


@dataclass
class TrainResult:
    """O que train() retorna e o que é persistido em disco ao final do treino."""

    model_name: str                 # "yolov8n", "ssd300_vgg16", "fasterrcnn_resnet50_fpn"
    dataset_dir: str
    config: dict
    epochs_trained: int
    train_time_s: float
    checkpoint_path: str
    history: list = field(default_factory=list)   # [{epoch, train_loss, val_loss, map50, ...}, ...]
    best_metrics: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d_%H-%M-%S"))

    def save(self, path: Optional[str] = None) -> str:
        """Salva o resultado como JSON. Se `path` não for informado,
        usa <output_dir>/<experiment_name>/train_result.json"""
        if path is None:
            path = str(Path(self.config.get("output_dir", "results"))
                        / self.config.get("experiment_name", "exp")
                        / "train_result.json")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)
        return path

    @staticmethod
    def load(path: str) -> "TrainResult":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return TrainResult(**data)


@dataclass
class EvalMetrics:
    """O que evaluate() retorna. Métricas de precisão + eficiência,
    para permitir o benchmark comparativo (precisão vs velocidade vs custo)."""

    model_name: str
    dataset_dir: str
    split: str
    map50: float
    map50_95: float
    precision: float
    recall: float
    f1: float
    fps: float
    inference_time_ms: float
    num_params: int
    per_class: dict = field(default_factory=dict)  # {"com_capacete": {"ap50": .., ...}, ...}
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d_%H-%M-%S"))

    def save(self, path: str) -> str:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)
        return path
