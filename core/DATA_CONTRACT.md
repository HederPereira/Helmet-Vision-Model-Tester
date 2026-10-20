# Contrato de dados

Todos os wrappers recebem o **mesmo** `dataset_dir`, com esta estrutura fixa
(anotações no formato YOLO-txt, que é o mais simples de converter para os
outros formatos):

```
dataset_dir/
├── images/
│   ├── train/   *.jpg | *.png
│   ├── val/
│   └── test/
├── labels/
│   ├── train/   *.txt   (um .txt por imagem, mesmo nome)
│   ├── val/
│   └── test/
└── classes.txt          (uma classe por linha, ordem = class_id)
```

Cada linha de um `.txt` em `labels/`:
```
<class_id> <x_center> <y_center> <width> <height>   # tudo normalizado [0,1]
```

- **YoloWrapper**: usa esse formato quase diretamente — só precisa gerar um
  `data.yaml` apontando para essas pastas.
- **SsdWrapper / FasterRcnnWrapper**: usam `data/yolo_txt_dataset.py`
  (a implementar na Fase 2), um `torch.utils.data.Dataset` que lê essa mesma
  estrutura e converte on-the-fly para o formato que o torchvision espera
  (bboxes em pixels absolutos, formato `{boxes, labels}`).

Manter um único formato de origem evita ter 3 cópias do dataset e garante
que treino/val/test sejam **exatamente** os mesmos exemplos para os 3 modelos
— condição necessária para o benchmark ser justo.
