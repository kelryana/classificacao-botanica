# 🌸 Classificação Botânica com Redes Neurais Convolucionais (Oxford Flowers-102)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/Torchvision-0.17%2B-red.svg)](https://pytorch.org/vision/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Auditado](https://img.shields.io/badge/Status-Auditado%20%26%20Corrigido-blue.svg)]()
[![Tests: 12 passed](https://img.shields.io/badge/Tests-12%20passed-success.svg)]()

> **Pipeline científico e auditado de Visão Computacional para categorização botânica de grão fino (*Fine-Grained Visual Categorization — FGVC*) no benchmark Oxford Flowers-102, empregando MobileNetV2, Transfer Learning, Fine-Tuning progressivo com controle estrito de normalização, testes estatísticos pareados, calibração probabilística e interpretabilidade visual.**

---

##  Visão Geral do Projeto

A classificação automatizada de espécimes vegetais constitui um dos desafios mais expressivos da visão computacional aplicada devido ao problema de **Granularidade Fina (*Fine-Grained Classification*)**:
- **Elevada Variabilidade Intra-Classe:** Indivíduos da mesma espécie vegetal apresentam divergências fenotípicas pronunciadas provocadas por senescência corolar, desidratação, insolação direta e ângulo de tomada óptica.
- **Sutil Variabilidade Inter-Classes:** Espécies filogeneticamente próximas ou morfologicamente convergentes compartilham geometrias radiais e padrões de coloração quase indistinguíveis em imagens bidimensionais.

O projeto adota o benchmark canônico **Oxford Flowers-102** (Nilsback & Zisserman, 2008), composto por $8.189$ imagens distribuídas em $102$ espécies botânicas, preservando estritamente as partições oficiais:
- **Treinamento:** $1.020$ imagens ($10$ amostras por classe).
- **Validação:** $1.020$ imagens ($10$ amostras por classe).
- **Teste:** $6.149$ imagens (avaliação densa para quantificação de incerteza).

---

##  Arquitetura do Pipeline em 6 Estágios

O pipeline científico é estruturado de forma modular e integrada:

```mermaid
flowchart LR
    E1["1. Aquisição & Splits<br/>(1.020 Tr / 1.020 Val / 6.149 Te)"] --> E2["2. Pré-processamento<br/>(RandomResizedCrop / CenterCrop 224)"]
    E2 --> E3["3. Extração Convolucional<br/>(MobileNetV2: Embeddings 1280-d)"]
    E3 --> E4["4. Treinamento 2 Estágios<br/>(Fase 1: MLP Head / Fase 2: Fine-Tuning)"]
    E4 --> E5["5. Avaliação & Métricas<br/>(McNemar, Bootstrap, ECE, Grad-CAM)"]
    E5 --> E6["6. Governança & Limitações<br/>(Análise de Erros & Lab-to-Field Gap)"]
```

1. **Estágio 1 — Aquisição de Dados:** Particionamento oficial e integridade dos rótulos botânicos (102 espécies).
2. **Estágio 2 — Pré-processamento e Aumento:** Invariância estocástica no treino (crop, flip, rotação de 20°, jitter de cor) e preservação de proporção no teste.
3. **Estágio 3 — Extração de Características:** Representações latentes de 1.280 dimensões via MobileNetV2 pré-treinada no ImageNet-1K e benchmarking clássico (KNN, Random Forest, Regressão Logística).
4. **Estágio 4 — Modelagem Supervisionada em Dois Estágios:**
   - *Fase 1:* Cabeça MLP densa com Dropout e BatchNorm, com backbone e estatísticas congelados.
   - *Fase 2:* Fine-Tuning progressivo dos blocos convolucionais finais (`features.16-18`) com taxas diferenciais.
5. **Estágio 5 — Avaliação Quantitativa, Interpretabilidade e Calibração:**
   - Acurácia Top-1, Top-5 e Macro F1 no teste completo ($6.149$ imagens).
   - Teste de McNemar com correção de Edwards e Bootstrap pareado ($B=1000$).
   - Diagrama de confiabilidade e métricas ECE, Brier Score e NLL.
   - Auditoria qualitativa via Grad-CAM na última camada convolucional.
6. **Estágio 6 — Governança e Limitações:** Análise dos principais gargalos de confusão entre pares de classes e discussão crítica das barreiras para aplicação em campo.

---

##  Validação de Software e Testes Unitários

Antes de qualquer execução experimental prolongada, o código foi submetido a uma suíte abrangente de testes de software em [`test_pipeline.py`](test_pipeline.py). Todos os 12 testes foram executados e aprovados:

| Módulo / Teste | Propósito da Validação | Resultado |
| :--- | :--- | :---: |
| `test_classification_metrics_perfect` | Valida Top-1, Top-5 e F1 com predições perfeitas em subsets de classes | **APROVADO** |
| `test_classification_metrics_imperfect` | Valida cálculo exato de Top-1 e Top-2 com erros controlados | **APROVADO** |
| `test_ece_perfect_calibration` | Confirma que predições perfeitamente calibradas resultam em ECE $\approx 0$ | **APROVADO** |
| `test_ece_edge_case_prob_one` | Testa se amostras com probabilidade $1,0$ são alocadas com segurança no último bin | **APROVADO** |
| `test_ece_empty_bins_handling` | Assegura que bins desabitados não recebam artificialmente acurácia $0\%$ | **APROVADO** |
| `test_contingency_table` | Valida contagens discordantes $b$ e $c$ para testes pareados | **APROVADO** |
| `test_mcnemar_known_values` | Valida a fórmula do $\chi^2_{\text{Edwards}}$ com valores analíticos conhecidos | **APROVADO** |
| `test_mcnemar_identical_models` | Verifica se modelos com predições idênticas retornam $\chi^2 = 0$ e $p = 1,0$ | **APROVADO** |
| `test_paired_bootstrap` | Valida que o Bootstrap pareado estima corretamente intervalos de diferença positiva | **APROVADO** |
| `test_flower_classifier_forward` | Valida dimensões de entrada $(B, 3, 224, 224)$ e saída $(B, 102)$ do modelo | **APROVADO** |
| `test_freeze_backbone_bn_eval_mode` | Garante que `running_mean` do BatchNorm não se altere durante a Fase 1 | **APROVADO** |
| `test_transforms_aspect_ratio_preservation` | Confirma redimensionamento isotrópico de imagens retangulares | **APROVADO** |

Para reproduzir os testes localmente:
```bash
pytest test_pipeline.py -v
```

---

## Ambiente, Infraestrutura e Dependências

- **Linguagem:** Python 3.10+ (testado em Python 3.12).
- **Framework Principal:** PyTorch 2.x e Torchvision.
- **Ecossistema Científico:** Scikit-Learn, SciPy, NumPy, Pandas, Matplotlib, Seaborn, UMAP-learn, Python-Docx.
- **Aceleração de Hardware Suportada:** GPU NVIDIA com suporte a CUDA e Precisão Mista Automática (AMP FP16), como a Tesla T4 (16 GB VRAM) disponível no Google Colab Pro.

As versões exatas e compatíveis estão fixadas no arquivo [`requirements.txt`](requirements.txt).

---

## Como Executar e Reproduzir

### Opção A: Execução no Google Colab Pro com GPU (Recomendado)
1. Abra o Google Colab Pro.
2. Carregue o notebook [`executar_colab.ipynb`](executar_colab.ipynb).
3. Selecione o ambiente de execução com acelerador GPU (ex: T4 GPU).
4. Execute as células em sequência para instalar dependências, rodar a suíte de testes, executar o pipeline completo e inspecionar as métricas persistidas.

### Opção B: Execução Local / Linha de Comando
Clone o repositório e crie o ambiente virtual:
```bash
git clone https://github.com/kelryana/classificacao-botanica.git
cd classificacao-botanica
git checkout revisao-metodologica

# Instalação das dependências
pip install -r requirements.txt

# Execução da suíte de testes
pytest test_pipeline.py -v

# Execução do pipeline
python pipeline_botanico.py --base-dir ./resultados_botanico --data-dir ./data --epochs-p1 8 --epochs-p2 4 --batch-size 32
```

---

## Protocolo de Avaliação Estatística e Calibração

O pipeline gera artefatos estruturados que permitem auditoria completa e independente:
- `metrics.json`: Dicionário contendo todas as métricas numéricas calculadas (Top-1, Top-5, Macro F1, Brier Score, NLL, ECE, $\chi^2_{\text{Edwards}}$, $p$-valor e intervalos Bootstrap).
- `predictions.npz`: Tensores NumPy contendo `y_true`, probabilidades Softmax e predições por classe de todos os modelos avaliados.
- `checkpoint_best_ft.pth`: Pesos do modelo neural ajustado na Fase 2.

---

## Estrutura do Repositório

```text
classificacao-botanica/
├── AUDITORIA_METODOLOGICA.md       # Relatório formal de auditoria e discrepâncias
├── README.md                       # Documentação técnica e metodológica (este arquivo)
├── requirements.txt                # Dependências fixadas e compatíveis
├── pipeline_botanico.py            # Pipeline completo de engenharia e modelagem
├── test_pipeline.py                # Suíte de 12 testes unitários (pytest)
└── executar_colab.ipynb            # Notebook executável e documentado para Google Colab Pro
```
