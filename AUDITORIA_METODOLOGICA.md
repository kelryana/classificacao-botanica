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

## 3. Plano de Correções

1. **Separação de Resultados Não Validados:** Arquivos anteriores retirados do repositório público (mantidos apenas localmente pela autora).
2. **Refatoração do Código (`pipeline_botanico.py`):**
   - Configuração centralizada (`Config` dataclass/dict) para seeds, hiperparâmetros, diretórios e flags.
   - Remoção de qualquer piso forçado (`max()`), deltas de subconjunto ou p-valores hardcoded.
   - Implementação de TTA consistente avaliada no conjunto de teste completo (ou congelamento e documentação clara).
   - Mapeamento botânico oficial: utilizar os 102 nomes de espécies reais; para o UMAP, plotar por classes individuais ou clusters de características, eliminando a atribuição artificial `i % len(FAMILIES)`.
   - Cálculo real do teste de McNemar pareado (com contagens $b, c$, correção de Edwards e valor-$p$ exato) para todas as comparações reportadas.
   - Bootstrap pareado da diferença de acurácia entre Fase 1 e Fase 2 ($B=1000$).
   - Correção do algoritmo de ECE (tratamento de limite superior $\le 1.0$, tratamento correto de bins vazios sem atribuição de acurácia zero). Adição de Brier Score e NLL.
   - Fixação de BatchNorm do backbone em `eval()` durante a Fase 1.
   - Transformação determinística para extração de embeddings de treino (`val_test_transforms`).
   - Padronização de resizing com preservação de aspect-ratio (`Resize(256)`, `CenterCrop(224)`) conforme especificação oficial do Torchvision MobileNetV2.
   - Persistência estruturada: salvar `predictions.npz` (probs, preds, targets), `metrics.json`, `history.json` e checkpoints do modelo.
3. **Bateria de Testes de Software (`test_pipeline.py`):**
   - Testar corretude do cálculo de ECE com vetores sintéticos (incluindo $p=1.0$ e bins vazios).
   - Testar cálculo de McNemar com matrizes de contingência conhecidas.
   - Testar cálculo de bootstrap pareado.
   - Testar integridade de dimensões dos modelos e transformações.
4. **Notebook Reproduzível para Colab Pro (`executar_colab.ipynb`):**
   - Notebook limpo e autocontido que conecta ao Google Drive para persistir os artefatos.
5. **Artigo Acadêmico e README Atualizados:**
   - Textos sóbrios, sem conclusões categóricas não comprovadas, com distinção clara entre hipóteses teóricas e evidências experimentais.
   - Se o treinamento em GPU não for executado em sessão ativa do Colab pelo agente, marcar o artigo como **"RASCUNHO PENDENTE DE EXECUÇÃO"**, fornecendo toda a infraestrutura pronta para execução pela autora.
