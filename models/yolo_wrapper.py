"""
Wrapper para YOLO via Ultralytics.

É o mais simples dos três porque o Ultralytics já expõe train/val/predict
de forma bem próxima ao que definimos na interface comum -- aqui o trabalho
é principalmente "traduzir" TrainConfig/dataset_dir para o que a lib espera,
e "traduzir" de volta os resultados para TrainResult/EvalMetrics.
"""

import time
import shutil
from pathlib import Path
from typing import List, Union

import yaml

from core.base_model import BaseDetectorWrapper
from core.schemas import TrainConfig, TrainResult, EvalMetrics, Detection


class YoloWrapper(BaseDetectorWrapper):

    model_name = "yolo"

    def __init__(self, num_classes: int, class_names: List[str], variant: str = "yolov8n.pt"):
        """
        variant: checkpoint base do Ultralytics, ex. "yolov8n.pt", "yolov8s.pt",
                 "yolo11n.pt". Trocar isso é o suficiente para comparar
                 variantes de tamanho dentro da própria família YOLO.
        """
        super().__init__(num_classes, class_names)
        self.variant = variant
        self.model_name = f"yolo_{Path(variant).stem}"  # ex: "yolo_yolov8n"

    def build(self, config: TrainConfig) -> None:
        from ultralytics import YOLO  # import local: só exige a lib se este wrapper for usado

        weights = self.variant if config.pretrained else self.variant.replace(".pt", ".yaml")
        self.model = YOLO(weights)

    # ------------------------------------------------------------------ #
    def _write_data_yaml(self, dataset_dir: Union[str, Path], out_path: Path) -> Path:
        """Gera o data.yaml que o Ultralytics espera, a partir do dataset_dir
        no formato do nosso DATA_CONTRACT.md"""
        dataset_dir = Path(dataset_dir).resolve()
        classes_file = dataset_dir / "classes.txt"
        names = [l.strip() for l in classes_file.read_text(encoding="utf-8").splitlines() if l.strip()]

        data_yaml = {
            "path": str(dataset_dir),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(names),
            "names": names,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data_yaml, f, allow_unicode=True)
        return out_path

    # ------------------------------------------------------------------ #
    def train(self, dataset_dir: Union[str, Path], config: TrainConfig) -> TrainResult:
        if self.model is None:
            self.build(config)

        run_dir = (Path(config.output_dir) / config.experiment_name).resolve()
        data_yaml = self._write_data_yaml(dataset_dir, run_dir / "data.yaml")

        t0 = time.time()
        results = self.model.train(
            data=str(data_yaml),
            epochs=config.epochs,
            imgsz=config.img_size,
            batch=config.batch_size,
            device=config.device,
            workers=config.num_workers,
            patience=config.patience,
            seed=config.seed,
            # `project` PRECISA ser absoluto: com um caminho relativo o Ultralytics
            # o reinterpreta como <runs_dir>/<task>/<project> (ver get_save_dir em
            # ultralytics/cfg/__init__.py) e grava em runs/detect/... em vez de aqui.
            project=str(run_dir.parent),
            name=run_dir.name,
            exist_ok=True,
            verbose=False,
        )
        train_time = time.time() - t0

        # Mesmo com `project` absoluto, não supomos onde o Ultralytics gravou:
        # perguntamos ao trainer (increment_path pode ter acrescentado sufixo).
        ul_run_dir = self._ultralytics_run_dir(default=run_dir)

        # Ultralytics salva um results.csv com o histórico por época dentro do run_dir
        history = self._parse_results_csv(ul_run_dir / "results.csv")

        checkpoint_path = self.save_checkpoint(run_dir / "weights" / "best.pt")

        best_metrics = {}
        if history:
            best_metrics = max(history, key=lambda h: h.get("metrics/mAP50(B)", 0))

        result = TrainResult(
            model_name=self.model_name,
            dataset_dir=str(dataset_dir),
            config=config.to_dict(),
            epochs_trained=len(history) if history else config.epochs,
            train_time_s=train_time,
            checkpoint_path=str(checkpoint_path),
            history=history,
            best_metrics=best_metrics,
        )
        result.save()
        return result

    @staticmethod
    def _parse_results_csv(csv_path: Path) -> list:
        if not csv_path.exists():
            return []
        import csv as csv_module
        with open(csv_path, "r") as f:
            reader = csv_module.DictReader(f)
            return [
                {k.strip(): _try_float(v) for k, v in row.items()}
                for row in reader
            ]

    # ------------------------------------------------------------------ #
    def evaluate(self, dataset_dir: Union[str, Path], split: str = "test") -> EvalMetrics:
        if self.model is None:
            raise RuntimeError(f"[{self.model_name}] carregue um checkpoint antes de avaliar.")

        # Caminho absoluto pelo mesmo motivo de train(): com `project` relativo
        # o Ultralytics gravaria em runs/detect/val, val-2, val-3...
        eval_dir = (Path("results") / "_eval" / f"{self.model_name}_{split}").resolve()
        data_yaml = self._write_data_yaml(dataset_dir, eval_dir / "data.yaml")

        t0 = time.time()
        metrics = self.model.val(
            data=str(data_yaml),
            split=split,
            project=str(eval_dir.parent),
            name=eval_dir.name,
            exist_ok=True,
            verbose=False,
        )
        elapsed = time.time() - t0

        # metrics.box.* vem do Ultralytics (objeto DetMetrics)
        n_images = max(1, getattr(metrics, "seen", 1))
        inference_time_ms = (elapsed / n_images) * 1000
        fps = 1000 / inference_time_ms if inference_time_ms > 0 else 0.0

        per_class = {}
        try:
            for i, name in enumerate(self.class_names):
                per_class[name] = {"ap50": float(metrics.box.ap50[i])}
        except Exception:
            pass  # se a versão do Ultralytics expuser diferente, não quebra o benchmark

        return EvalMetrics(
            model_name=self.model_name,
            dataset_dir=str(dataset_dir),
            split=split,
            map50=float(metrics.box.map50),
            map50_95=float(metrics.box.map),
            precision=float(metrics.box.mp),
            recall=float(metrics.box.mr),
            f1=_safe_f1(float(metrics.box.mp), float(metrics.box.mr)),
            fps=fps,
            inference_time_ms=inference_time_ms,
            num_params=self.count_parameters(),
            per_class=per_class,
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

        results = self.model.predict(
            source=str(source), conf=conf_threshold, save=False, verbose=False
        )

        detections: List[Detection] = []
        for r in results:  # um item por frame/imagem
            names = r.names
            for box in r.boxes:
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                cls_id = int(box.cls[0])
                detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        score=float(box.conf[0]),
                        class_id=cls_id,
                        class_name=names.get(cls_id, str(cls_id)),
                    )
                )
            if save_output:
                out = Path(output_path) if output_path else Path("results/_predictions") / Path(source).name
                out.parent.mkdir(parents=True, exist_ok=True)
                r.save(filename=str(out))

        return detections

    # ------------------------------------------------------------------ #
    def _ultralytics_run_dir(self, default: Path) -> Path:
        """Diretório onde o Ultralytics realmente gravou o run.

        Nunca supor que é `<output_dir>/<experiment_name>`: quem manda é o
        `trainer.save_dir`, que já passou por `increment_path`.
        """
        trainer = getattr(self.model, "trainer", None)
        save_dir = getattr(trainer, "save_dir", None) if trainer is not None else None
        return Path(save_dir) if save_dir else Path(default)

    def _trained_weights_path(self) -> Union[Path, None]:
        """Pesos produzidos pelo treino desta instância (`best.pt`, senão `last.pt`).

        Não usar `self.model.ckpt_path` para isso: o Ultralytics NÃO o atualiza
        após o treino, então ele continua apontando para o checkpoint base
        (ex. yolov8n.pt) e salvá-lo gravaria o modelo não treinado.
        """
        trainer = getattr(self.model, "trainer", None)
        if trainer is None:
            return None
        for attr in ("best", "last"):
            candidate = getattr(trainer, attr, None)
            if candidate and Path(candidate).is_file():
                return Path(candidate)
        return None

    def save_checkpoint(self, path: Union[str, Path]) -> str:
        if self.model is None:
            raise RuntimeError(f"[{self.model_name}] build() precisa ser chamado antes.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        source = self._trained_weights_path()
        if source is not None:
            if source.resolve() != path.resolve():
                shutil.copy(source, path)
        else:
            # Sem treino nesta instância (ex. após load_checkpoint): serializa
            # o modelo que está em memória.
            self.model.save(str(path))
        return str(path)

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        from ultralytics import YOLO
        self.model = YOLO(str(path))


def _try_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def _safe_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
