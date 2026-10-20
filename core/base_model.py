"""
Interface comum (padrão Adapter/Wrapper) para arquiteturas de detecção.

Cada framework (Ultralytics, TorchVision) é radicalmente diferente por baixo,
mas o restante do sistema (Trainer, Evaluator, UI) só conhece esses 5 métodos.
Isso é o que permite trocar de modelo trocando uma linha de config.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union

from core.schemas import TrainConfig, TrainResult, EvalMetrics, Detection


class BaseDetectorWrapper(ABC):
    """Todo wrapper de modelo (YoloWrapper, SsdWrapper, FasterRcnnWrapper)
    deve herdar desta classe e implementar os métodos abaixo."""

    #: nome curto usado em logs, nomes de arquivo e na UI
    model_name: str = "base"

    def __init__(self, num_classes: int, class_names: List[str]):
        self.num_classes = num_classes
        self.class_names = class_names
        self.model = None  # instância concreta do framework, criada em build()

    # ------------------------------------------------------------------ #
    # Ciclo de vida do modelo
    # ------------------------------------------------------------------ #
    @abstractmethod
    def build(self, config: TrainConfig) -> None:
        """Instancia o modelo (arquitetura + pesos pré-treinados se aplicável).
        Deve deixar `self.model` pronto para treino ou inferência."""
        raise NotImplementedError

    @abstractmethod
    def train(self, dataset_dir: Union[str, Path], config: TrainConfig) -> TrainResult:
        """Treina o modelo sobre o dataset em `dataset_dir`.

        `dataset_dir` segue SEMPRE a mesma estrutura de diretório
        independente do modelo (ver core/data_contract.md) — é
        responsabilidade do wrapper converter para o formato que o
        framework espera (ex: data.yaml para Ultralytics, listas de
        (image, target) para torchvision).

        Retorna um TrainResult e já o persiste em disco (config.output_dir).
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, dataset_dir: Union[str, Path], split: str = "test") -> EvalMetrics:
        """Avalia o modelo já treinado (self.model) sobre o split indicado.
        Retorna métricas padronizadas (mAP, precision, recall, FPS, etc.)."""
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        source: Union[str, Path],
        conf_threshold: float = 0.25,
        save_output: bool = True,
        output_path: Union[str, Path, None] = None,
    ) -> List[Detection]:
        """Roda inferência em uma imagem ou vídeo (`source`).
        Se `save_output=True`, salva a imagem/vídeo anotado em `output_path`.
        Retorna a lista de detecções (do último frame, se for vídeo, ou
        agregada — ver implementação de cada wrapper)."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Persistência
    # ------------------------------------------------------------------ #
    @abstractmethod
    def save_checkpoint(self, path: Union[str, Path]) -> str:
        """Salva os pesos do modelo treinado. Retorna o path final salvo."""
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Carrega pesos de um checkpoint salvo anteriormente, deixando
        self.model pronto para evaluate()/predict()."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Utilitário comum a todos (não precisa ser sobrescrito)
    # ------------------------------------------------------------------ #
    def count_parameters(self) -> int:
        """Implementação default assumindo um nn.Module do PyTorch por baixo
        (válido para os 3 wrappers atuais). Pode ser sobrescrita se preciso."""
        if self.model is None:
            raise RuntimeError(f"[{self.model_name}] build() precisa ser chamado antes.")
        underlying = getattr(self.model, "model", self.model)  # Ultralytics guarda em .model
        return sum(p.numel() for p in underlying.parameters())

    def export_metrics(self, metrics: EvalMetrics, output_dir: Union[str, Path]) -> str:
        """Salva um EvalMetrics em <output_dir>/<model_name>_<split>_metrics.json"""
        output_dir = Path(output_dir)
        path = output_dir / f"{self.model_name}_{metrics.split}_metrics.json"
        return metrics.save(str(path))
