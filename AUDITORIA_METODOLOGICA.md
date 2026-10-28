# Relatório de Auditoria Metodológica e Divergências de Artefatos

**Projeto:** Classificação Botânica (Oxford Flowers-102 com MobileNetV2)  
**Data da Auditoria:** 28 de outubro de 2026  
**Branch de Auditoria:** `revisao-metodologica`  
**Commit Base:** `69ea386` (remotes/origin/main)  

---

## 1. Visão Geral da Auditoria

Esta auditoria detalha as falhas metodológicas, inconsistências lógicas e discrepâncias entre o código (`pipeline_botanico.py`), a documentação (`README.md`), o relatório gerado (`relatorio_botanico(1).docx`) e as figuras correspondentes.

Nenhum artefato pré-existente possui rastreabilidade com checkpoints de pesos gravados, predições por imagem (`.npy`), arquivos de métricas (`.json`), sementes isoladas ou logs completos de treino.

---

## 2. Inventário de Divergências e Inconsistências Identificadas

| Item | Implementação no Código Legado (`pipeline_botanico.py`) | Descrição no `README.md` | Descrição no `relatorio_botanico.docx` / Figuras | Status Metodológico |
| :--- | :--- | :--- | :--- | :--- |
| **A. Test-Time Augmentation (TTA)** | Avaliação apenas com flip horizontal em subconjunto aleatório de 1.200 imagens do teste; ganho artificial forçado com `delta_tta = max(0.65, acc_tta_sub - acc_std_sub)`; acurácia final calculada como soma do delta de subconjunto com a acurácia total. | Afirma implementação de política "5-crop / Flip", acurácia de 84,44% e $p < 10^{-30}$. | Texto no DOCX afirma ganho empírico com TTA de +0,65% e acurácia de 84,44%. | **INVÁLIDO / FABRICADO**. O ganho artificial `max(0.65, ...)` e a projeção de subconjunto invalidam qualquer alegação de TTA. Não existia política 5-crop no código. |
| **B. Taxonomia e Projeção UMAP** | Linhas 138-142: `CLASS_FAMILIES = [FAMILIES[i % len(FAMILIES)] for i in range(102)]`. Famílias atribuídas por módulo sequencial às 102 classes. | Figura 4 e texto alegam agrupamentos naturais biologicamente consistentes por macro-famílias botânicas (*Asteraceae*, *Orchidaceae*, etc.). | Figura 4 colorida por macro-famílias e texto afirmando preservação evolutiva. | **INVÁLIDO / FABRICADO**. As famílias não correspondem à taxonomia biológica real das classes do Flowers-102. Nenhuma inferência biológica pode ser feita. |
| **C. Testes Estatísticos e $p$-valores** | Linhas 1243-1245: Na tabela, para KNN, Random Forest e Regressão Logística, a coluna de $p$-valor continha a string fixa `"p < 10⁻¹⁰"`. McNemar só foi calculado entre Fase 1 e Fase 2. | Tabela de Síntese exibe $p < 10^{-10}$ para KNN, RF e LogReg como se tivessem sido calculados formalmente. | Tabela 1 contém $p < 10^{-10}$ hardcoded. | **INVÁLIDO / HARDCODED**. Valores literais não calculados a partir de predições pareadas. |
| **D. Intervalos de Confiança (Bootstrap)** | Bootstrap reamostrou o array de predições da Fase 2 para estimar IC de 95% da acurácia final. Para o ganho do fine-tuning ($\Delta_{\text{FT}}$), nenhum bootstrap pareado foi executado. | Relata IC 95% [82,91%, 84,65%] e faz inferências amplas de significância. | Reporta IC do modelo final. | **INCOMPLETO / CONFUSO**. O IC de predições em split único não captura variabilidade entre sementes de treinamento. Falta bootstrap pareado da diferença de acurácia entre as fases. |
| **E. Calibração e Cálculo de ECE** | `np.digitize(max_probs, bins) - 1` com `linspace(0, 1, 11)`. Probabilidade exata 1.0 cai no índice de bin 10 (fora dos 10 bins se não houver clip); bins vazios tinham acurácia atribuída como 0.0 na plotagem, distorcendo visualmente a barra. | Afirma ECE = 3,42% e calibração dita "excelente". | Afirma calibração excelente sem métricas de suporte como Brier score ou Negative Log-Likelihood (NLL). | **FALHA DE IMPLEMENTAÇÃO**. Risco de descarte/overflow de amostras com confiança 1.0 e distorção visual em bins desabitados. |
| **F. BatchNorm na Fase 1** | Chamada `model.train()` colocava a rede inteira em modo treino. Embora os pesos do backbone estivessem com `requires_grad=False`, as camadas `BatchNorm2d` do backbone poderiam atualizar estatísticas correntes (`running_mean` e `running_var`). | Afirma que o backbone permaneceu "estritamente congelado". | Afirma congelamento total dos extratores na Fase 1. | **INCONSISTÊNCIA DE CONTROLE**. Se a intenção é congelamento estrito, o backbone precisa manter `eval()` ou congelar explicitamente os buffers de BatchNorm. |
| **G. Redimensionamento de Imagens** | Utilizava `Resize((224, 224))` sem preservação de razão de aspecto para teste/validação, distorcendo corolas e proporções morfológicas. | Argumenta que o pipeline evita distorções de aspecto através de `RandomResizedCrop` no treino, mas ignora a distorção no teste. | Não menciona que a avaliação realizou estiramento não-isotrópico. | **SUB-ÓTIMO**. O protocolo padrão do Torchvision para MobileNetV2 prescreve `Resize(256)` seguido de `CenterCrop(224)`. |
| **H. Extração de Embeddings para Modelos Clássicos** | A função `extract_embeddings` utilizava o `train_loader`, cuja transformação possuía `RandomResizedCrop`, `RandomRotation` e `ColorJitter` ativos. Cada imagem de treino foi representada por um crop aleatório único. | Afirma que modelos clássicos avaliaram as características latentes estáticas da MobileNetV2. | Não explicita que o treino dos modelos clássicos recebeu features aumentadas aleatoriamente. | **METODOLOGIA CONFUSA**. Para benchmarking de representações estáticas de backbone, deve-se usar transformação determinística de avaliação ou documentar explicitamente o protocolo estocástico. |
| **I. Rastreabilidade e Auditoria** | Nenhum arquivo `.npy`, `.json` de métricas, checkpoints `.pth` ou arquivos de log da execução estavam versionados ou salvos em disco. Apenas arquivos gráficos e o `.docx` estavam presentes. | Apresenta números com precisão de duas casas decimais como fatos consumados. | Apresenta números como fatos consumados. | **NÃO AUDITÁVEL**. Impossível verificar se as execuções anteriores foram completas, simuladas ou parciais. |
| **J. Afirmações Extrapoladas** | Não há código implementando quantização INT8, exportação ONNX, TensorRT, latência em smartphone (<15ms), monitoramento de drift com KS-test ou validação molecular (DNA Barcoding / LAMP). | README e DOCX apresentam esses tópicos de forma que pode induzir o leitor a crer que foram implementados no projeto. | Idem. | **EXTRAPOLAÇÃO NÃO SUPORTADA**. Devem ser reclassificados explicitamente como discussão teórica e trabalhos futuros. |

---

## 3. Ações Concluídas e Status Atual da Auditoria

Todas as não-conformidades identificadas foram integralmente corrigidas, testadas e documentadas:

1. **Separação e Blindagem de Dados:**
   - Todos os artefatos históricos não auditados (figuras antigas, vídeos MP4 e relatórios antigos) foram retirados do repositório público e mantidos exclusivamente no computador local da autora.
   - O `.gitignore` foi configurado para impedir o envio acidental de arquivos pessoais, artigos, caches e diretórios históricos ao GitHub.

2. **Refatoração Completa do Código (`pipeline_botanico.py`):**
   - **TTA Íntegro:** Eliminação de qualquer piso forçado (`max()`) e de projeções parciais. O TTA (imagem normal + reflexão horizontal) é calculado sobre todas as 6.149 imagens de teste.
   - **Taxonomia Real:** Eliminação da atribuição arbitrária de famílias (`i % len(FAMILIES)`). Projeção UMAP mapeia as classes botânicas reais.
   - **Testes Estatísticos Pareados:** Cálculo estrito de $\chi^2_{\text{Edwards}}$ com $df=1$ e $p$-valor assintótico para todas as comparações pareadas, sem valores literais digitados.
   - **Bootstrap Pareado:** Implementação de reamostragem com reposição ($B=1000$) para o ganho real entre a Fase 1 e o Fine-Tuning ($\Delta_{\text{FT}}$).
   - **Calibração Robusta:** Correção dos limites do ECE ($p \le 1,0$), proteção de bins desabitados (sem viés artificial de 0% de acurácia) e integração de Brier Score e NLL.
   - **Controle de BatchNorm:** `model.backbone.eval()` mantido explicitamente durante a Fase 1, congelando médias e variâncias correntes.
   - **Preservação de Proporção:** Validação e teste padronizados com `Resize(256)` e `CenterCrop(224)`.
   - **Extração Determinística:** Modelos clássicos ajustados com transformações sem perturbação estocástica.
   - **Persistência Completa:** O script salva `predictions.npz`, `metrics.json`, `history.json` e checkpoint `.pth`.

3. **Verificação por Testes de Software (`test_pipeline.py`):**
   - Implementação de 12 testes unitários cobrindo métricas, invariância de BatchNorm, ECE em casos-limite, McNemar analítico e Bootstrap pareado.
   - **Resultado:** **12 de 12 testes aprovados com 100% de sucesso** via `pytest`.

4. **Status das Figuras Técnicas (F01 a F11):**
   - As figuras anteriores foram invalidadas porque utilizavam agrupamentos falsos e deltas artificiais.
   - O código para renderizar as **11 novas figuras técnicas em 300 DPI** está totalmente concluído na função `generate_all_plots` do `pipeline_botanico.py`.
   - Para manter total honestidade e conformidade científica (sem simular dados que não foram treinados), as novas figuras serão gravadas diretamente na pasta `resultados_botanico/plots/` durante a execução do notebook no Colab Pro, refletindo os pesos e predições reais obtidos na GPU.

5. **Status do Artigo Acadêmico:**
   - O artigo completo e formatado conforme a norma ABNT NBR 6023:2018 foi gerado em formato editável (`.docx`) e compilado em `.pdf`.
   - O documento reflete o estado auditado do protocolo e foi classificado internamente como rascunho metodológico com infraestrutura pronta, preservado como arquivo pessoal local da autora.

6. **Prontidão para Execução no Google Colab Pro:**
   - O notebook [`executar_colab.ipynb`](executar_colab.ipynb) está finalizado, com diagnóstico de GPU, instalação de dependências e chamada do pipeline auditado, permitindo à autora consolidar os resultados empíricos em aproximadamente 4 a 6 minutos na GPU Tesla T4.
