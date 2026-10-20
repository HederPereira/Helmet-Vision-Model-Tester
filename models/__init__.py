"""
Factory de modelos: mapeia um nome (string, ex. vindo da UI ou de um config.yaml)
para o wrapper concreto correspondente. É o único lugar que "conhece" todos
os wrappers -- o resto do sistema (trainer, evaluator, UI) só chama get_model().
"""

from core.base_model import BaseDetectorWrapper

_REGISTRY = {
    "yolo": "models.yolo_wrapper.YoloWrapper",
    "ssd": "models.ssd_wrapper.SsdWrapper",
    "faster_rcnn": "models.faster_rcnn_wrapper.FasterRcnnWrapper",
}


def get_model(name: str, num_classes: int, class_names: list, **kwargs) -> BaseDetectorWrapper:
    """
    Exemplo de uso:
        model = get_model("yolo", num_classes=2, class_names=["com_capacete", "sem_capacete"])
        model = get_model("faster_rcnn", num_classes=2, class_names=[...])
    """
    key = name.lower().strip()
    if key not in _REGISTRY:
        raise ValueError(f"Modelo '{name}' não registrado. Opções: {list(_REGISTRY.keys())}")

    module_path, class_name = _REGISTRY[key].rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    wrapper_cls = getattr(module, class_name)
    return wrapper_cls(num_classes=num_classes, class_names=class_names, **kwargs)
