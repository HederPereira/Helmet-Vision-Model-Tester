#!/usr/bin/env python
"""Nível 3: ciclo completo de um wrapper — build, train, evaluate, predict,
save/load de checkpoint.

A pergunta que este script responde é "o código roda de ponta a ponta e devolve
os schemas certos?", não "o modelo aprendeu?". Use o dataset sintético.

Uso:
    .venv/bin/python scripts/test_wrapper.py yolo --device cpu
    .venv/bin/python scripts/test_wrapper.py ssd --epochs 1
    .venv/bin/python scripts/test_wrapper.py faster_rcnn --batch-size 2
    .venv/bin/python scripts/test_wrapper.py all
"""

import _bootstrap  # noqa: F401

import argparse
import hashlib
import json
import sys
from pathlib import Path

from core.schemas import EvalMetrics, TrainResult
from core.schemas import TrainConfig
from models import get_model

CLASSES = ["com_capacete", "sem_capacete"]
MODELOS = ("yolo", "ssd", "faster_rcnn")


def md5(path) -> str | None:
    path = Path(path)
    return hashlib.md5(path.read_bytes()).hexdigest() if path.is_file() else None


def checar_checkpoint_yolo(run_dir: Path, resultado: TrainResult) -> None:
    """Regressão do bug corrigido em 20/10/2026: o checkpoint registrado chegava
    a ser uma cópia byte-a-byte do modelo base, não do treinado — e de forma
    silenciosa. Ver 'Pontos de atenção' nº 1 no guia."""
    ck = resultado.checkpoint_path
    base = next((p for p in (Path("yolov8n.pt"), Path("yolov8n.pt").resolve()) if p.is_file()), None)
    print(f"  checkpoint registrado : {ck}  md5={md5(ck)}")
    if base:
        print(f"  modelo base           : {base}  md5={md5(base)}")
        if md5(ck) and md5(ck) == md5(base):
            print("  >> BUG: o checkpoint salvo é o modelo NÃO TREINADO")
    reais = sorted(Path("runs").rglob("*.pt")) if Path("runs").exists() else []
    if reais:
        print(f"  pesos que o Ultralytics gravou de fato: {[str(p) for p in reais[:4]]}")
    if not resultado.history:
        print("  >> ATENÇÃO: history vazio — o wrapper não achou o results.csv do Ultralytics")


def testar(nome: str, args) -> dict | None:
    print(f"\n{'=' * 64}\n== {nome}\n{'=' * 64}")
    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
        device=args.device,
        patience=args.epochs + 1,     # sem early stopping num teste tão curto
        output_dir="results",
        experiment_name=f"smoke_{nome}",
    )
    run_dir = Path("results") / cfg.experiment_name

    modelo = get_model(nome, num_classes=len(CLASSES), class_names=CLASSES)
    modelo.build(cfg)
    print(f"parâmetros: {modelo.count_parameters():,}")

    resultado = modelo.train(args.dataset, cfg)
    assert isinstance(resultado, TrainResult), "train() deve retornar TrainResult"
    print(f"treino    : {resultado.epochs_trained} época(s) em {resultado.train_time_s:.1f}s")
    print(f"history   : {len(resultado.history)} registro(s)")
    if resultado.history:
        print(f"  primeiro: {resultado.history[0]}")
    assert (run_dir / "train_result.json").is_file(), "train() deve persistir o TrainResult"

    if nome == "yolo":
        checar_checkpoint_yolo(run_dir, resultado)

    metricas = modelo.evaluate(args.dataset, split="test")
    assert isinstance(metricas, EvalMetrics), "evaluate() deve retornar EvalMetrics"
    assert metricas.model_name == modelo.model_name and metricas.split == "test"
    assert metricas.num_params > 0 and metricas.fps > 0
    print(f"avaliação : mAP50={metricas.map50:.4f}  mAP50-95={metricas.map50_95:.4f}  "
          f"FPS={metricas.fps:.2f}  {metricas.inference_time_ms:.1f}ms/img")
    if metricas.per_class:
        print(f"  por classe: {metricas.per_class}")
    print(f"  json: {modelo.export_metrics(metricas, run_dir)}")

    imagem = next(Path(args.dataset, "images", "test").glob("*.jpg"))
    deteccoes = modelo.predict(imagem, conf_threshold=0.05,
                               output_path=run_dir / "pred.jpg")
    print(f"predict   : {len(deteccoes)} detecção(ões) em {imagem.name} -> {run_dir}/pred.jpg")

    # round-trip de checkpoint: salva, recria do zero, recarrega e reavalia.
    # É o que prova que o checkpoint registrado é mesmo o modelo treinado.
    caminho = modelo.save_checkpoint(run_dir / "manual.pt")
    recarregado = get_model(nome, num_classes=len(CLASSES), class_names=CLASSES)
    if nome != "yolo":
        recarregado.build(cfg)          # torchvision: fixa o device antes do load
    recarregado.load_checkpoint(caminho)
    m2 = recarregado.evaluate(args.dataset, split="test")
    assert abs(m2.map50 - metricas.map50) < 1e-6, (metricas.map50, m2.map50)
    print("checkpoint: save/load reproduz a mesma métrica")

    return {
        "modelo": modelo.model_name,
        "map50": metricas.map50,
        "map50_95": metricas.map50_95,
        "fps": metricas.fps,
        "params": metricas.num_params,
        "treino_s": resultado.train_time_s,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("modelo", choices=(*MODELOS, "all"))
    ap.add_argument("--dataset", default="datasets/dummy_helmet")
    ap.add_argument("--device", default="cpu", help="cpu | cuda:0")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--img-size", type=int, default=320)
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    if not Path(args.dataset).is_dir():
        print(f"dataset não encontrado: {args.dataset}\n"
              f"gere um com: .venv/bin/python scripts/make_dummy_dataset.py", file=sys.stderr)
        return 1

    alvos = MODELOS if args.modelo == "all" else (args.modelo,)
    linhas = [testar(nome, args) for nome in alvos]

    print(f"\n{'=' * 78}")
    print(f"{'modelo':<26}{'mAP50':>9}{'mAP50-95':>10}{'FPS':>8}{'params':>14}{'treino(s)':>11}")
    for r in linhas:
        print(f"{r['modelo']:<26}{r['map50']:>9.4f}{r['map50_95']:>10.4f}"
              f"{r['fps']:>8.2f}{r['params']:>14,}{r['treino_s']:>11.1f}")
    Path("results").mkdir(exist_ok=True)
    Path("results/smoke_summary.json").write_text(json.dumps(linhas, indent=2), encoding="utf-8")
    print("\nresumo salvo em results/smoke_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
