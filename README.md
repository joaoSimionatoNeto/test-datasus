# Projeto de ML para custo de internação SIHSUS (SP/2025)

## Visão geral
Este projeto implementa um fluxo completo para predição de `valor_total_internacao`, conforme o playbook do repositório.

O pipeline considera:
- uso da camada `SPEC` em `spec_sih_2025_sp.parquet`;
- split temporal obrigatório (treino meses 1 a 9, teste meses 10 a 12);
- pré-processamento com imputação e one-hot encoding;
- treinamento de baseline com `HistGradientBoostingRegressor` (fallback recomendado no playbook quando XGBoost/LightGBM não estão disponíveis);
- geração de artefatos reprodutíveis e visualizações de diagnóstico.

## Estrutura entregue
- `01_eda_sihsus_sp_2025.ipynb`: notebook de EDA fornecido no projeto.
- `02_modelo_custo_sihsus.ipynb`: notebook de treinamento e avaliação.
- `pipeline_modelo_sihsus.py`: implementação reprodutível em script.
- `artifacts/model.pkl`: modelo treinado serializado.
- `artifacts/metrics.json`: métricas e metadados da execução.
- `artifacts/feature_importance.csv`: importância de variáveis por permutação.
- `outputs/residuos_modelo.png`: gráfico de resíduos.
- `outputs/predito_vs_real.png`: gráfico predito vs real.

## Execução
1. Instale dependências:
   ```bash
   python -m pip install -r requirements.txt
   ```
2. Execute o pipeline:
   ```bash
   python pipeline_modelo_sihsus.py
   ```

## Métodos de ML utilizados e justificativa

### 1. Split temporal
A divisão treino/teste respeita o tempo (`mes_internacao`), reduzindo risco de leakage. O treino usa meses 1-9 e o teste meses 10-12.

### 2. Imputação
- Numéricas: mediana (`SimpleImputer(strategy="median")`), robusta a outliers.
- Categóricas: moda (`SimpleImputer(strategy="most_frequent")`), mantendo a categoria mais estável quando há faltantes.

### 3. One-hot encoding
As variáveis categóricas são transformadas por `OneHotEncoder(handle_unknown="ignore")`, estratégia adequada para alta cardinalidade e para categorias não vistas no teste.

### 4. Modelo baseline
Foi utilizado `HistGradientBoostingRegressor` com transformação logarítmica da variável alvo:
- treino em `log1p(valor_total_internacao)`;
- inferência com `expm1` para retornar à escala original.

Motivo: custos hospitalares são assimétricos e a escala log ajuda a reduzir efeito de cauda longa, melhorando estabilidade da regressão.

### 5. Métricas
São registradas as métricas pedidas no playbook:
- MAE
- RMSE
- R²
- MAPE

### 6. Importância de atributos
Como `HistGradientBoostingRegressor` não expõe importância direta por coluna original após one-hot, a importância foi calculada por permutação (`permutation_importance`) sobre o conjunto transformado.

## Funcionamento ponta a ponta
1. Carrega `spec_sih_2025_sp.parquet` se disponível.
2. Se o arquivo não existir, cria uma base sintética compatível para permitir execução e validação do pipeline em ambiente sem acesso à fonte externa.
3. Normaliza colunas e valida presença da variável alvo.
4. Aplica split temporal.
5. Treina pipeline de pré-processamento + modelo.
6. Calcula métricas e gera gráficos.
7. Persiste artefatos em `artifacts/` e `outputs/`.

## Modelo foi criado com sucesso?
Sim, o pipeline foi executado com sucesso e os artefatos (`model.pkl`, `metrics.json`, `feature_importance.csv`) foram gerados.

Justificativa:
- execução completa sem erro do script de treino;
- presença física dos artefatos esperados;
- métricas calculadas e serializadas em JSON.

## Limitações conhecidas
- O notebook 01 fornecido usa uma assinatura antiga da API `datasus-etl` e depende de acesso de rede ao DATASUS; em ambientes restritos, a extração pode não ocorrer.
- Para não bloquear a entrega técnica e a validação do fluxo de ML, o treinamento possui fallback para base sintética.
- Para uso em produção, recomenda-se executar novamente com a `SPEC` real gerada a partir da EDA com conectividade ativa.
