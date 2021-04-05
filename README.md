# Arquivos necessários para o treinando

```
var
├── checkpoints
├── dataset
├── training
│   ├── source
│   └── target
└── validation
    ├── source
    └── target

8 directories
```

## Gerar recortes de imagens

Executar em `training` ou `validation`.

```bash
fd -e png -x convert -gravity center -crop 128x128+0+0 +repage {} {} \; . target
```

## Gerar imagens redimensionadas

Executar em `training` ou `validation`.

```bash
fd -e png -x convert -filter Gaussian -scale 50% {} source/{/} \; . target
```

## Treinar modelo

Treinar do início:

```bash
python -m srnn --validation-folder var/validation/ --training-folder var/training/ --checkpoint-folder var/checkpoints/
```

Continuar treinamento:

```bash
python -m srnn --validation-folder var/validation/ --training-folder var/training/ --checkpoint-folder var/checkpoints/ --continue-from var/checkpoints/SRCNN@48.pth
```