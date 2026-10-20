#!/usr/bin/env python
"""Nível 0: confere o ambiente e a fumaça do código (imports, factory, schemas).

Uso: .venv/bin/python scripts/check_env.py
"""

import _bootstrap  # noqa: F401  (sys.path + cwd)

import sys


def check_ambiente() -> bool:
    print("== Ambiente ==")
    print(f"python      {sys.version.split()[0]}")
    print(f"venv        {'sim' if sys.prefix != sys.base_prefix else 'NÃO — você está fora da venv'}")
    print(f"prefix      {sys.prefix}")

    ok = True
    for nome in ("torch", "torchvision", "ultralytics", "torchmetrics",
                 "pycocotools", "yaml", "numpy", "PIL"):
        try:
            mod = __import__(nome)
            print(f"  {nome:<14} {getattr(mod, '__version__', 'ok')}")
        except ImportError:
            print(f"  {nome:<14} FALTANDO")
            ok = False

    import torch
    print(f"\ntorch CUDA compilada: {torch.version.cuda}")
    print(f"cuda.is_available():  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)} | capability {torch.cuda.get_device_capability(0)}")
        a = torch.randn(256, 256, device="cuda") @ torch.randn(256, 256, device="cuda")
        torch.cuda.synchronize()
        print(f"  matmul na GPU: OK {tuple(a.shape)}")
    else:
        print("  -> rode os testes com device='cpu' (veja a seção de GPU no guia)")
    return ok


def check_codigo() -> bool:
    print("\n== Nível 0: imports, factory e schemas ==")
    from core.base_model import BaseDetectorWrapper
    from core.schemas import TrainConfig, TrainResult
    from models import get_model

    ok = True

    try:
        BaseDetectorWrapper(2, ["a", "b"])
        print("  FALHA  BaseDetectorWrapper deveria ser abstrata")
        ok = False
    except TypeError:
        print("  OK     BaseDetectorWrapper é abstrata")

    for nome in ("yolo", "ssd", "faster_rcnn"):
        w = get_model(nome, num_classes=2, class_names=["com_capacete", "sem_capacete"])
        assert isinstance(w, BaseDetectorWrapper)
        print(f"  OK     get_model({nome!r}) -> {type(w).__name__} (model_name={w.model_name})")

    try:
        get_model("detr", 2, ["a", "b"])
        print("  FALHA  factory aceitou modelo inexistente")
        ok = False
    except ValueError:
        print("  OK     factory rejeita nome desconhecido")

    cfg = TrainConfig(epochs=1, output_dir="results", experiment_name="_check_env")
    tr = TrainResult("x", "d", cfg.to_dict(), 1, 1.0, "ck.pt")
    path = tr.save()
    assert TrainResult.load(path).model_name == "x"
    print("  OK     TrainResult salva e recarrega")

    import shutil
    shutil.rmtree("results/_check_env", ignore_errors=True)
    return ok


if __name__ == "__main__":
    ok = check_ambiente() & check_codigo()
    print("\n" + ("TUDO OK" if ok else "HÁ PENDÊNCIAS — veja as linhas acima"))
    sys.exit(0 if ok else 1)
