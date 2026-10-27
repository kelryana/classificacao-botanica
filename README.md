# Pipeline de Engenharia de Visão Computacional: Classificação Botânica (Flowers-102)

## Descrição do Projeto
Este projeto implementa um pipeline completo de Visão Computacional focado na classificação de espécies botânicas utilizando o dataset **Oxford 102 Flowers**. O script aborda todo o ciclo de vida do desenvolvimento de um modelo robusto, passando pela exploração de dados, extração de features, treinamento, e validação avançada com interpretabilidade computacional.

A solução combina técnicas estado-da-arte, incluindo **Transfer Learning**, **Progressive Fine-Tuning**, **Benchmarking Multiparadigma**, análise de **Interpretabilidade Visual (Grad-CAM)** e **Calibração Probabilística**.

## Principais Funcionalidades e Técnicas
- **Transfer Learning e Progressive Fine-Tuning**: Adaptação de modelos pré-treinados para a tarefa específica com ajuste fino progressivo em camadas profundas.
- **Benchmarking Multiparadigma**: Comparação de performance entre modelos de Deep Learning (Redes Neurais Convolucionais) e Algoritmos Clássicos de Machine Learning (K-Nearest Neighbors, Regressão Logística, Random Forest).
- **Projeção de Embeddings (UMAP)**: Redução de dimensionalidade para visualização no espaço latente das características (features) extraídas.
- **Interpretabilidade e XAI (Explainable AI)**: Implementação de Grad-CAM autêntico para gerar mapas de ativação/calor, permitindo entender quais regiões da imagem o modelo foca para tomar decisões.
- **Calibração Probabilística e Governança de Drift**: Análise detalhada da confiabilidade das predições, crucial para o deployment seguro de modelos preditivos.
- **Geração Automatizada de Relatórios**: Exportação de todas as métricas, tabelas e gráficos analisados para um documento consolidado em formato Microsoft Word (`.docx`).

## Tecnologias Utilizadas
- **Linguagem Principal**: Python 3
- **Deep Learning**: PyTorch 2.x, Torchvision
- **Machine Learning & Métricas**: Scikit-Learn (KNN, Logistic Regression, Random Forest, accuracy, F1-score)
- **Visualização e Redução de Dimensionalidade**: Matplotlib, Seaborn, UMAP
- **Automação de Relatórios**: Python-Docx
- **Infraestrutura Otimizada**: Suporte nativo ao backend gráfico `Agg` (Headless) e aproveitamento de GPU (Google Colab T4 / CUDA) com Mixed Precision (`torch.amp.autocast`).

## Estrutura de Artefatos Gerados
A execução do pipeline `pipeline_botanico.py` resulta na geração automática de análises visuais avançadas e arquivos estruturados para fácil entendimento dos stakeholders. Os arquivos são salvos, por padrão, na pasta base do projeto ou em `resultados_botanico`:

- **Imagens e Gráficos de Análise:**
  - `F01_amostras_dataset.png`: Amostras do conjunto de dados visualizadas.
  - `F02_data_augmentation.png`: Demonstração de técnicas de Data Augmentation.
  - `F03_hierarquia_features.png`: Visualização de mapas de características da rede.
  - `F04_umap_embeddings.png`: Projeção UMAP 2D das embeddings geradas pela CNN.
  - `F05_comparativo_classificadores.png`: Benchmarking dos algoritmos clássicos versus a CNN extraidora de features.
  - `F06_curvas_aprendizado_fase1.png`: Curvas da fase inicial (Transfer Learning/Warm-up).
  - `F07_curvas_consolidadas_finetuning.png`: Curvas do treinamento com Progressive Fine-Tuning completo.
  - `F08_matriz_confusao.png`: Avaliação detalhada do acerto e confusão de predições por classe.
  - `F09_gradcam_interpretabilidade.png`: Mapas de calor de interpretabilidade visual (Grad-CAM).
  - `F10_calibracao_confiabilidade.png`: Diagnóstico de calibração probabilística do modelo.
  - `F11_analise_classes_erros.png`: Estudo das classes com as maiores taxas de erro.

- **Documentação e Relatório Final:**
  - `relatorio_botanico.docx`: Documento detalhado e formatado automaticamente em MS Word englobando as métricas, tabelas de benchmark e explicações visuais de cada experimento.
  - `analise_computacional_dados_visuais-v3.pdf`: Documentação em formato PDF com a base teórica e análise completa do experimento.
  - `video_explicativo.mp4`: Material de vídeo instrutivo do projeto.

## Como Executar

**1. Instale as dependências:**
Certifique-se de possuir Python 3.x, além do suporte a CUDA instalado (recomendado) para aproveitar processamento em placa de vídeo.
```bash
pip install torch torchvision scikit-learn matplotlib seaborn pandas umap-learn python-docx
```

**2. Execute o Pipeline Principal:**
```bash
python pipeline_botanico.py
```

O script criará um diretório e salvará todos os gráficos de avaliação e o documento de texto consolidados no final do processamento. Em caso de ausência da biblioteca UMAP, ele lidará com os imports de maneira adaptável.
