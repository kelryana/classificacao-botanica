# 🌸 Classificação Botânica com Redes Neurais Convolucionais (Oxford Flowers-102)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/Torchvision-0.17%2B-red.svg)](https://pytorch.org/vision/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Auditado](https://img.shields.io/badge/Status-Auditado%20%26%20Corrigido-blue.svg)]()
[![Tests: 16 passed](https://img.shields.io/badge/Tests-16%20passed-success.svg)]()

> **Pipeline científico auditado e reproduzível de Visão Computacional para categorização botânica de grão fino (*Fine-Grained Visual Categorization — FGVC*) no benchmark Oxford Flowers-102, empregando MobileNetV2, Transfer Learning, Fine-Tuning seletivo com controle estrito de `BatchNorm2d`, benchmarking linear convexo, testes estatísticos pareados, calibração probabilística e interpretabilidade visual.**

---

##  Visão Geral do Projeto

A classificação automatizada de espécimes vegetais insere-se no domínio de **Categorização Visual de Grão Fino (*Fine-Grained Visual Categorization — FGVC*)**, caracterizado por:
- **Elevada Variabilidade Intra-Classe:** Indivíduos da mesma espécie vegetal apresentam divergências fenotípicas pronunciadas provocadas por estágio fenológico/ontogenético, desidratação, iluminação natural e ângulo de tomada óptica.
- **Sutil Variabilidade Inter-Classes:** Espécies filogeneticamente próximas ou morfologicamente convergentes compartilham simetrias florais e espectros cromáticos fortemente sobrepostos em imagens bidimensionais.

O projeto adota o benchmark canônico **Oxford Flowers-102** (Nilsback & Zisserman, 2008), composto por $8.189$ imagens distribuídas em $102$ espécies botânicas, preservando estritamente as partições oficiais:
- **Treinamento (`train`):** $1.020$ imagens ($10$ amostras por classe).
- **Validação (`val`):** $1.020$ imagens ($10$ amostras por classe).
- **Teste (`test`):** $6.149$ imagens ($20$ a $248$ amostras por classe).

##  Configuração Efetiva e Hiperparâmetros do Código (`pipeline_botanico.py`)

| Componente | Configuração Exata Executada no Código |
| :--- | :--- |
| **Backbone & Pesos** | `torchvision.models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)` ($1.280$-d após `adaptive_avg_pool2d`) |
| **Cabeça Classificadora (MLP)** | `Dropout(0.3) -> Linear(1280, 512) -> BatchNorm1d(512) -> ReLU -> Dropout(0.2) -> Linear(512, 102)` ($709.222$ parâmetros) |
| **Transformações de Treino** | `RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.85, 1.15), antialias=True)`, `RandomHorizontalFlip(0.5)`, `RandomRotation(20)`, `ColorJitter(0.15, 0.15, 0.15)`, `Normalize(ImageNet)` |
| **Transformações de Val/Teste e Extração** | `Resize(256, antialias=True)` (interpolação bilinear preservando razão de aspecto) + `CenterCrop(224)` + `Normalize(ImageNet)` |
| **Fase 1 (Cabeça MLP)** | $8$ épocas, `backbone` congelado com `model.backbone.eval()` ativo (preservando `running_mean` e `running_var` do `BatchNorm2d`), `AdamW(lr=1e-3, weight_decay=1e-4)`, sem scheduler, seleção por menor `val_loss` |
| **Fase 2 (Fine-Tuning Seletivo)** | $4$ épocas, descongelamento seletivo de `features.16`, `features.17` e `features.18` ($1.915.302$ parâmetros treináveis totais), `AdamW` com taxas diferenciais `lr_backbone=1e-4` e `lr_head=5e-4`, `weight_decay=1e-4`, sem scheduler *Cosine Annealing*, seleção por menor `val_loss` |
| **TTA (Test-Time Augmentation)** | Média aritmética simples das probabilidades Softmax entre a imagem original e sua reflexão horizontal especular ($P_{\text{TTA}}(x) = 0,5 P(x) + 0,5 P(\text{Flip}(x))$) avaliada exatamente uma vez sobre todas as $6.149$ imagens de teste |
| **Semente e Precisão** | `seed = 42` (`cudnn.deterministic = True`, `cudnn.benchmark = False`), `batch_size_train = 32`, `batch_size_eval = 64`, AMP (`autocast("cuda")` + `GradScaler`) |

---

## Resultados Verificados no Conjunto de Teste ($N = 6.149$, Execução Colab Pro `commit a7a15d3`)

A auditoria completa que reconcilia os números preliminares descartados (`79,88% → 83,79%) com a execução refatorada no Google Colab Pro (`81,27% → 84,19%`, GPU NVIDIA Tesla T4)

| Modelo / Paradigma | Acertos ($k / 6.149$) | Top-1 (%) | Top-5 (%) | Macro F1 (%) | $\Delta$ Top-1 vs. Fase 1 | Teste de McNemar vs. Fase 1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **KNN ($k=5$, euclidiana)** | $4.176 / 6.149$ | `67,91%` | N/R* | `68,23%` | `-13,35 p.p.` | Log truncado ($b, c$ N/R*) |
| **Random Forest ($100$, `max_depth=20`)** | $3.812 / 6.149$ | `61,99%` | N/R* | `58,21%` | `-19,27 p.p.` | Log truncado ($b, c$ N/R*) |
| **Regressão Logística L2 ($C=1,0$)** | $5.296 / 6.149$ | **`86,13%`** | N/R* | **`85,66%`** | `+4,86 p.p.` | Log truncado ($b, c$ N/R*) |
| **Cabeça MLP — Fase 1 (`backbone.eval()`)** | $4.997 / 6.149$ | `81,27%` | `95,06%` | `80,72%` | Referência | Referência |
| **MobileNetV2 — Fase 2 (Fine-Tuning)** | $5.177 / 6.149$ | `84,19%` | `95,77%` | `83,91%` | `+2,93 p.p.` (não arred.) / `+2,92 p.p.` (arred.) | $b=457, c=277, \chi^2=43,65, p=3,92 \times 10^{-11}$ |
| **Fase 2 + TTA (Flip Horizontal Completo)** | $5.215 / 6.149$ | `84,81%` | `95,98%` | `84,49%` | `+3,55 p.p.` (não arred.) | $+38$ acertos líquidos sobre Fase 2 |

- **Intervalo de Confiança Bootstrap 95% (Modelo Final Fase 2):** `[83,27%, 85,09%]`
- **Bootstrap Pareado $\Delta(\text{Fase 2} - \text{Fase 1})$:** média `+2,94 p.p.`, IC 95% `[+2,11 p.p., +3,84 p.p.]`
- **Calibração Probabilística (Fase 2 no Teste):** $\text{ECE} = 11,70\%$, $\text{Brier Score} = 0,2482$, $\text{NLL} = 0,6675$ (perfil de subconfiança nas faixas intermediárias de probabilidade).

---

## Validação de Software e Testes Unitários (`test_pipeline.py`)

A suíte automatizada em [`test_pipeline.py`](test_pipeline.py) conta com **16 testes unitários** e um teste sintético de fumaça (`--smoke-test`) de ponta a ponta:

<<<<<<< HEAD
| Módulo / Teste | Propósito da Validação | Status Esperado |
| :--- | :--- | :---: |
| `test_classification_metrics_perfect` | Valida Top-1, Top-5, Macro F1 e contagens exatas com predições perfeitas | **APROVADO** |
| `test_classification_metrics_imperfect` | Valida cálculo exato de Top-1, Top-2 e contagens com erros controlados | **APROVADO** |
| `test_ece_perfect_calibration` | Confirma que predições perfeitamente calibradas resultam em $\text{ECE} \approx 0$ | **APROVADO** |
| `test_ece_edge_case_prob_one` | Testa se amostras com probabilidade $1,0$ são alocadas no último bin $[0,9, 1,0]$ | **APROVADO** |
| `test_ece_empty_bins_handling` | Assegura que bins vazios recebam `None` e não acurácia artificial $0\%$ | **APROVADO** |
| `test_contingency_table` | Valida contagens discordantes pareadas $b$ e $c$ | **APROVADO** |
| `test_mcnemar_known_values` | Valida $\chi^2_{\text{Edwards}}$ e $p$-valor (incluindo $b=457, c=277 \Rightarrow \chi^2=43,6526$) | **APROVADO** |
| `test_mcnemar_identical_models` | Verifica se modelos idênticos retornam $\chi^2 = 0$ e $p = 1,0$ | **APROVADO** |
| `test_paired_bootstrap` | Valida que o Bootstrap pareado estima corretamente intervalos de diferença | **APROVADO** |
| `test_flower_classifier_forward` | Valida dimensões de entrada $(B, 3, 224, 224)$ e saída $(B, 102)$ | **APROVADO** |
| `test_freeze_backbone_bn_eval_mode` | Garante que `running_mean` e `running_var` do `BatchNorm2d` não mudem na Fase 1 | **APROVADO** |
| `test_deterministic_embedding_extraction` | Confirma determinismo bit-a-bit na extração de embeddings em `eval()` | **APROVADO** |
| `test_tta_full_averaging_no_floor` | Verifica média de probabilidades do TTA no loader completo sem piso forçado | **APROVADO** |
| `test_transforms_aspect_ratio_preservation` | Confirma redimensionamento isotrópico de imagens retangulares | **APROVADO** |
| `test_flower_names_completeness` | Verifica as 102 classes em `FLOWER_NAMES` e `FLOWER_SCIENTIFIC_NAMES` | **APROVADO** |
| `test_recalculate_metrics_from_npz` | Valida recálculo completo de métricas e testes estatísticos via `predictions.npz` | **APROVADO** |
=======
| Módulo / Teste | Propósito da Validação 
| :--- | :--- | 
| `test_classification_metrics_perfect` | Valida Top-1, Top-5 e F1 com predições perfeitas em subsets de classes | 
| `test_classification_metrics_imperfect` | Valida cálculo exato de Top-1 e Top-2 com erros controlados | 
| `test_ece_perfect_calibration` | Confirma que predições perfeitamente calibradas resultam em ECE $\approx 0$ | 
| `test_ece_edge_case_prob_one` | Testa se amostras com probabilidade $1,0$ são alocadas com segurança no último bin | 
| `test_ece_empty_bins_handling` | Assegura que bins desabitados não recebam artificialmente acurácia $0\%$ | 
| `test_contingency_table` | Valida contagens discordantes $b$ e $c$ para testes pareados | 
| `test_mcnemar_known_values` | Valida a fórmula do $\chi^2_{\text{Edwards}}$ com valores analíticos conhecidos | 
| `test_mcnemar_identical_models` | Verifica se modelos com predições idênticas retornam $\chi^2 = 0$ e $p = 1,0$ | 
| `test_paired_bootstrap` | Valida que o Bootstrap pareado estima corretamente intervalos de diferença positiva | 
| `test_flower_classifier_forward` | Valida dimensões de entrada $(B, 3, 224, 224)$ e saída $(B, 102)$ do modelo | 
| `test_freeze_backbone_bn_eval_mode` | Garante que `running_mean` do BatchNorm não se altere durante a Fase 1 | 
| `test_transforms_aspect_ratio_preservation` | Confirma redimensionamento isotrópico de imagens retangulares | 
>>>>>>> refs/remotes/origin/main

Para executar os testes e o smoke test localmente:
```bash
./.venv/bin/pytest test_pipeline.py -v
./.venv/bin/python pipeline_botanico.py --smoke-test
```

---

## Como Executar e Reproduzir no Google Colab Pro ou Localmente

### Opção A: Execução no Google Colab Pro com GPU e Persistência no Google Drive
1. Abra o notebook [`executar_colab.ipynb`](executar_colab.ipynb) no Google Colab Pro com acelerador GPU habilitado.
2. Execute as células em sequência para registrar a GPU real da sessão, montar o Google Drive, rodar os 16 testes unitários + smoke test, treinar o pipeline completo com `--drive-backup-dir` e verificar os resultados via `--verify-only`.

<<<<<<< HEAD
### Opção B: Linha de Comando
=======
As versões exatas e compatíveis estão fixadas no arquivo [`requirements.txt`](requirements.txt).

---

## Como Executar e Reproduzir

### Opção A: Link direto do Google Colab para você executar, salvar, fazer um copia ou alterar da sua maneira

1. Acesse o Link [`flores_classificacao`](https://colab.research.google.com/drive/198RGDFIjSfcrm6H8Y6c6zC0U27ZNHwX9?usp=sharing)
3. OU Carregue o notebook [`executar_colab.ipynb`](executar_colab.ipynb).
4. Selecione o ambiente de execução com acelerador GPU (ex: T4 GPU).
5. Execute as células em sequência para instalar dependências, rodar a suíte de testes, executar o pipeline completo e inspecionar as métricas persistidas.

### Opção b: Execução Local / Linha de Comando (Atenção, exige processamento dedicado, faça apenas se for escalável)
Clone o repositório e crie o ambiente virtual:
>>>>>>> refs/remotes/origin/main
```bash
pip install -r requirements.txt
pytest test_pipeline.py -v
python pipeline_botanico.py --smoke-test

# Execução completa do treinamento e geração de artefatos
python pipeline_botanico.py \
  --base-dir ./resultados_botanico \
  --data-dir ./data \
  --epochs-p1 8 \
  --epochs-p2 4 \
  --batch-size 32 \
  --umap-split train_val \
  --umap-max-samples 1500

# Recálculo e auditoria independente a partir de predictions.npz
python pipeline_botanico.py --verify-only ./resultados_botanico/predictions.npz

```

---

<<<<<<< HEAD
=======
## Protocolo de Avaliação Estatística e Calibração

O pipeline gera artefatos estruturados que permitem auditoria completa e independente:
- `metrics.json`: Dicionário contendo todas as métricas numéricas calculadas (Top-1, Top-5, Macro F1, Brier Score, NLL, ECE, $\chi^2_{\text{Edwards}}$, $p$-valor e intervalos Bootstrap).
- `predictions.npz`: Tensores NumPy contendo `y_true`, probabilidades Softmax e predições por classe de todos os modelos avaliados.
- `checkpoint_best_ft.pth`: Pesos do modelo neural ajustado na Fase 2.

---

>>>>>>> refs/remotes/origin/main
## Estrutura do Repositório

```text
classificacao-botanica/
<<<<<<< HEAD
├── README.md                             # Documentação técnica e guia de reprodutibilidade
├── requirements.txt                      # Dependências do projeto
├── pipeline_botanico.py                  # Pipeline principal de treinamento, avaliação e auditoria
├── test_pipeline.py                      # Suíte de 16 testes unitários automatizados (pytest)
├── executar_colab.ipynb                  # Notebook preparado para execução no Google Colab Pro
└── resultados_botanico/
    └── plots/                            # 11 figuras técnicas em alta resolução (300 DPI, F01..F11.png)
=======
├── README.md                       # Documentação técnica e metodológica (este arquivo)
├── requirements.txt                # Dependências fixadas e compatíveis
├── pipeline_botanico.py            # Pipeline completo de engenharia e modelagem
├── test_pipeline.py                # Suíte de 12 testes unitários (pytest)
└── executar_colab.ipynb            # Notebook executável e documentado para Google Colab Pro
>>>>>>> refs/remotes/origin/main
```
