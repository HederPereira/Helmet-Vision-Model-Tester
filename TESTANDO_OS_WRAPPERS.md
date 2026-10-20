# Testando os wrappers

Valida que `YoloWrapper`, `SsdWrapper` e `FasterRcnnWrapper` respeitam a mesma
interface (`BaseDetectorWrapper`) e devolvem os mesmos schemas
(`TrainResult` / `EvalMetrics`) a partir do **mesmo** `dataset_dir` — que é a
condição para o benchmark comparar os três de forma justa.

## Começo rápido

```bash
cd "/home/labvip/capacete/Helmet Vision Model Tester"

.venv/bin/python scripts/check_env.py                      # nível 0
.venv/bin/python scripts/make_dummy_dataset.py             # nível 1
.venv/bin/python scripts/check_contract.py                 # nível 2
.venv/bin/python scripts/test_wrapper.py all --device cpu  # nível 3
```

Rode nessa ordem — cada nível só faz sentido se o anterior passou. Os scripts
saem com código 0 quando passam e se viram sozinhos quanto a diretório
(entram na raiz do projeto por conta própria).

| script | nível | o que valida | tempo |
|--------|-------|--------------|-------|
| `check_env.py` | 0 | venv, versões, CUDA, imports, factory, schemas | segundos |
| `make_dummy_dataset.py` | 1 | gera dataset sintético no formato do contrato | segundos |
| `check_contract.py` | 2 | conversão YOLO-txt → torchvision (`YoloTxtDataset`) | segundos |
| `test_wrapper.py` | 3 | ciclo completo: build → train → evaluate → predict → checkpoint | minutos |

`test_wrapper.py` aceita `yolo`, `ssd`, `faster_rcnn` ou `all`, mais
`--device`, `--epochs`, `--batch-size`, `--img-size`, `--dataset`.

### Última execução: 20/10/2026, GPU (RTX 3080), 2 épocas, dataset sintético

Níveis 0, 1 e 2 passaram inteiros. Nível 3 — os três wrappers completam o ciclo,
incluindo o round-trip de checkpoint (salvar → recriar do zero → recarregar →
reavaliar com a mesma métrica):

| modelo | mAP50 | mAP50-95 | params | treino | veredito |
|--------|-------|----------|--------|--------|----------|
| yolov8n | 0,1003 | 0,0288 | 3.011.238 | 6,0 s | ✅ ciclo completo |
| ssd300_vgg16 | 0,5076 | 0,3151 | 23.879.570 | 1,0 s | ✅ ciclo completo |
| fasterrcnn_resnet50_fpn | 0,7886 | 0,4742 | 41.304.286 | 2,5 s | ✅ ciclo completo |

A coluna FPS foi omitida de propósito: ela **não é comparável** entre os
wrappers hoje (ver Ponto de atenção nº 3). Os valores de mAP não significam nada
além de "o caminho inteiro executou" — são 12 imagens sintéticas e 2 épocas.
Nível 4 (dataset real) segue pendente de dataset; a GPU voltou a funcionar.

## Ambiente

As dependências ficam numa venv **dentro do projeto** (`.venv/`), isolada dos
envs conda e de qualquer outro projeto da máquina.

```bash
# criar (uma vez) — Python 3.12 do conda como base: as wheels de
# torch/ultralytics têm melhor cobertura em 3.12 que no 3.13 do sistema
/home/labvip/miniconda3/bin/python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install torch torchvision        # wheel default do PyPI já traz CUDA
.venv/bin/pip install -r requirements.txt
```

Para usar: `source .venv/bin/activate` (sai com `deactivate`), ou chame
`.venv/bin/python` direto, como nos comandos acima. No VS Code,
`Ctrl+Shift+P` → "Python: Select Interpreter" → `.venv/bin/python`.

Instalado em 18/09/2026 (Python 3.12.8, ~5,9 GB): torch 2.14.0+cu130,
torchvision 0.29.0+cu130, ultralytics 8.4.155, torchmetrics 1.9.0,
pycocotools 2.0.11, numpy 2.5.3, PyYAML 6.0.3, Pillow 12.3.0.
`check_env.py` imprime tudo isso.

### Se a GPU sumir (`cuda.is_available() == False`)

Já aconteceu nesta máquina (18/09/2026; voltou ao normal desde então).
Sintoma: `torch.cuda.is_available()` retorna `False` e **não é problema da venv** —
`cuInit(0)` devolve `999` (`CUDA_ERROR_UNKNOWN`) para qualquer processo, mesmo
com `nvidia-smi` normal e os módulos do kernel na versão certa (580.95.05).
Na ocasião, os conda envs `rover`, `tfGpu`, `tfGpuCbam` e `deepfake-env`
também viam 0 GPUs — sinal de que o problema é da máquina, não do projeto.

É o sintoma clássico de `nvidia_uvm` em estado inconsistente (acontece após
suspender/retomar, ou após atualizar o driver com a sessão gráfica aberta):

```bash
sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm
.venv/bin/python -c "import torch; print(torch.cuda.is_available())"
```

Se o `rmmod` disser que o módulo está em uso, feche os processos com a GPU
(`nvidia-smi --query-compute-apps=pid,process_name --format=csv`); se nem
assim, reiniciar resolve.

Enquanto não voltar, use `--device cpu`: os níveis 0 a 3 rodam em CPU (mais
devagar); só o nível 4 (dataset real) exige GPU de verdade.

## O que observar

### Nível 2 é o mais perigoso

Se a conversão YOLO → torchvision estiver errada, SSD e Faster R-CNN treinam
com caixas erradas e a comparação com o YOLO fica inválida **sem erro nenhum**.
Por isso `check_contract.py` não se contenta com asserts: ele desenha as caixas
convertidas de volta sobre as imagens em `results/_contract_check/`.
**Abra essas imagens.** Números passando em assert não provam que o eixo não
está invertido. Repita este nível quando trocar para o dataset real.

### Nível 3 — checklist por wrapper

- `history` **não** vazio, com uma entrada por época
- `train_result.json` gravado em `results/smoke_<modelo>/`
- `EvalMetrics` com `num_params > 0`, `fps > 0`, nada `nan`
- imagem anotada em `results/smoke_<modelo>/pred.jpg`
- `save/load reproduz a mesma métrica` (SSD e Faster R-CNN)
- YOLO: `data.yaml` gerado com `path` absoluto e `names` vindos de `classes.txt`;
  o script audita o checkpoint automaticamente (ver Ponto de atenção nº 1)

**Critério de aprovação:** a tabela final do `test_wrapper.py all` sai com as
três linhas preenchidas, sem exceção e sem `nan`. Os valores em si não importam
no dataset sintético — ele existe para exercitar o caminho do código em
segundos, não para o modelo aprender.

## Nível 4 — dataset real

Só depois que o nível 3 passar inteiro:

1. Converta o dataset de capacetes para a estrutura do
   [core/DATA_CONTRACT.md](core/DATA_CONTRACT.md).
2. Rode `check_contract.py <dataset_dir>` **de novo**, incluindo a conferência
   visual — é o passo que pega erro de conversão.
3. Treine com `TrainConfig` de verdade: `epochs=50..100`, `img_size=640`,
   `batch_size` conforme couber nos 10 GB da 3080 (YOLO ~16, SSD ~8,
   Faster R-CNN ~4), `patience=10`.

Mantenha `seed=42` e o **mesmo** `dataset_dir` para os três modelos.

## Pontos de atenção

Achados reais, confirmados em execução. Nenhum impede os testes de rodar, mas
todos afetam números que forem para publicação.

### 1. ✅ CORRIGIDO — `YoloWrapper` gravava o checkpoint errado

Corrigido em 20/10/2026. **Causa raiz:** o Ultralytics só respeita `project`
quando ele é um caminho **absoluto**; com um caminho relativo ele reinterpreta
como `<runs_dir>/<task>/<project>` (ver `get_save_dir` em
`ultralytics/cfg/__init__.py`). O wrapper passava `"results"`, então o treino ia
para `runs/detect/results/<exp>/` enquanto o wrapper procurava em
`results/<exp>/` — não achava o `results.csv` (→ `history` vazio) nem o
`best.pt`, e acabava copiando `self.model.ckpt_path`, que o Ultralytics **não
atualiza** após o treino, gravando o modelo **não treinado** como se fosse o
resultado.

O que mudou em [models/yolo_wrapper.py](models/yolo_wrapper.py):

- `train()` passa `project` absoluto (`run_dir.parent` resolvido), então o
  Ultralytics grava direto em `results/<exp>/`;
- o diretório do run é lido de `self.model.trainer.save_dir`, não suposto;
- `save_checkpoint()` copia de `trainer.best` / `trainer.last` (novo helper
  `_trained_weights_path()`) e, quando não houve treino nesta instância,
  serializa o modelo em memória com `YOLO.save()` — nunca mais de `ckpt_path`;
- `evaluate()` também passa `project` absoluto, em `results/_eval/<modelo>_<split>/`,
  no lugar do antigo `results/_tmp_eval/` + `runs/detect/val`, `val-2`, `val-3`…

Validado: `history` preenchido, md5 do checkpoint diferente do modelo base, e
recarregar o checkpoint num processo novo reproduz o mAP50 exato do treino
(0,128201 nos dois). `runs/` não é mais criado. O round-trip de checkpoint
passou a rodar para os três wrappers no `test_wrapper.py`.

### 2. `precision`/`recall`/`f1` não são comparáveis entre os wrappers

SSD e Faster R-CNN preenchem `precision` com o próprio `map_50` e `f1` com
`0.0` (há `TODO` explícito no código); o YOLO usa `metrics.box.mp/mr` de
verdade. Compare por ora só `map50`, `map50_95`, `fps` e `num_params`.

### 3. 🔴 `fps` / `inference_time_ms` não são comparáveis entre os wrappers

Os dois lados medem coisas diferentes:

- **YOLO** ([models/yolo_wrapper.py](models/yolo_wrapper.py)) cronometra a
  chamada `model.val()` **inteira** — varredura do dataset, cálculo de métricas,
  geração dos gráficos — e divide pelo número de imagens.
- **SSD / Faster R-CNN** ([core/torchvision_common.py](core/torchvision_common.py))
  cronometram **só o forward pass** dentro do laço.

O erro é de ordens de magnitude. Na última bateria, para o mesmo modelo YOLO, o
Ultralytics reportou **1,8 ms/imagem de inferência** enquanto o wrapper gravou
**856,4 ms/imagem** (FPS 1,17) — o custo fixo do `val()` dominou, porque o split
de teste tem só 4 imagens.

Isso invalida justamente o eixo "velocidade" do benchmark. Correção sugerida:
usar o dicionário `metrics.speed` que o próprio Ultralytics devolve
(`{'preprocess', 'inference', 'postprocess'}`, em ms/imagem) e somar as partes
equivalentes ao que `evaluate_map` mede, em vez do relógio em volta do `val()`.

### 4. `load_checkpoint()` dos wrappers torchvision assume `cuda:0`

Se `self.model is None`, ele chama `build(TrainConfig(pretrained=False))`, que
usa o device default `"cuda:0"`. Em máquina sem GPU isso quebra. Chame
`build(cfg)` com o device desejado antes de `load_checkpoint()` — é o que o
`test_wrapper.py` faz.

### 5. Imagens sem anotação

`YoloTxtDataset` devolve `boxes` com shape `(0, 4)` quando o `.txt` está vazio
ou ausente. Faster R-CNN lida bem; o SSD pode levantar exceção no cálculo de
loss. Se o dataset real tiver imagens de fundo puro, teste isso explicitamente.

### 6. Caminhos relativos ao CWD

`YoloWrapper.evaluate` escreve `results/_tmp_eval/data.yaml` e os `predict()`
caem em `results/_predictions/`, sempre relativos ao diretório de onde o Python
foi chamado. Os scripts em `scripts/` já entram na raiz do projeto sozinhos; se
você chamar os wrappers direto do seu próprio código, rode da raiz.

### 7. `config/` e `ui/` estão vazios

São as fases seguintes (carregar `TrainConfig` de YAML e a interface). Nada nos
testes acima depende deles.

## Erros comuns

| erro | causa / solução |
|------|-----------------|
| `'MAP' metric requires pycocotools` | backend COCO do torchmetrics; já está no `requirements.txt` — `pip install pycocotools` |
| `No module named 'core'` / `'models'` | rodando fora da raiz do projeto |
| `CUDA unknown error` | driver da máquina, não o código — veja a seção da GPU |
| `CUDA out of memory` | reduza `--batch-size` (Faster R-CNN é o mais faminto) ou `--img-size` |
| DataLoader lento no dataset sintético | `--num-workers 0`: com 12 imagens, criar processos custa mais que ler |
| Ultralytics baixando `yolov8n.pt` | normal na primeira execução (~6 MB), depois fica em cache |
| `'SsdWrapper' object has no attribute '_device'` | `evaluate()`/`predict()` sem `build()` antes — ver Ponto de atenção nº 4 |
