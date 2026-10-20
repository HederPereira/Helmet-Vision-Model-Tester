"""Coloca a raiz do projeto no sys.path e no CWD.

Os wrappers usam imports absolutos de pacote (`core.`, `data.`, `models.`) e
gravam em caminhos relativos (`results/`, `runs/`). Importar este módulo no topo
de cada script deixa os dois corretos, independente de onde você chamou o script.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
