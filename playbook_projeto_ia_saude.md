# Playbook — ML Aplicada à Saúde (SIHSUS)  
**Projeto:** Análise e Predição de Custos de Internação no SIHSUS (SP/2025)  
**Objetivo geral:** construir uma base analítica e um arcabouço metodológico para gerar insights sobre internações e preparar variáveis para predição do valor total de internação.  
**Objetivos específicos:** estruturar o pipeline de dados (SOR/SOT/SPEC), enriquecer campos relacionais (CID‑10 e referências), conduzir EDA voltada à saúde e definir um baseline de modelagem supervisionada.

---

## 1) Contexto e escopo
- **Domínio:** saúde pública brasileira (SIHSUS – internações hospitalares).  
- **Recorte:** Estado de São Paulo, ano de 2025.  
- **Problema:** prever o **valor_total_internacao** a partir de atributos do paciente e do episódio assistencial.  
- **Restrições:** alta cardinalidade, assimetria dos custos, risco de *data leakage* (exigir validação temporal).

---

## 2) Ferramentas e dependências
### 2.1. Biblioteca principal (datasus-etl )
Utilize a bilbioteca datasus-etl  para obter os dados do datasus, documentação disponível no site: https://www.datasus-etl.com.br/pt/


### 2.2. Stack sugerida
- **pandas**, **numpy**  
- **scikit-learn** (pipeline, métricas)  
- **xgboost** ou **lightgbm** (baseline de boosting)  
- **matplotlib**, **seaborn**  

---

## 3) Arquitetura de dados (SOR / SOT / SPEC)
A biblioteca `datasus-etl` já executa internamente as etapas de:
- **Download** do FTP do DATASUS (SOR)
- **Conversão** DBC → DBF → DuckDB (SOT)
- **Enriquecimento** com IBGE (municípios) e CID-10 (SPEC)

Portanto, a separação explícita SOR/SOT é desnecessária: o pipeline entrega diretamente a camada **SPEC** (parquet particionado, enriquecido e pronto para modelagem). O `spec_sih_2025_sp.parquet` é gerado ao final do notebook de EDA.

---

## 4) Notebook 01 — EDA (Exploração e Preparação)
O notebook 01 - EDA já foi criado e está localizado neste repositório, ele possui o nome `01_eda_sihsus_sp_2025.ipynb` execute ele e use seus resultados como base para a construção do modelo

---

## 5) Notebook 02 — Treinamento do Modelo

**Nome sugerido:** `02_modelo_custo_sihsus.ipynb`

### 5.1. Carregamento
- Ler `spec_sih_2025_sp.parquet`

### 5.2. Split temporal (evitar leakage)
- Treino: meses 1–9  
- Teste: meses 10–12  

### 5.3. Pré-processamento
- Imputação:
  - Numéricas → mediana  
  - Categóricas → moda  
- **One-hot encoding** para alta cardinalidade  

### 5.4. Modelos base (baseline)
- **XGBoost** ou **LightGBM**  
- (Fallback se não disponível) **HistGradientBoostingRegressor**  

### 5.5. Testes de assertividade (métricas)
Calcular:
- **MAE**  
- **RMSE**  
- **R²**  
- **MAPE**  

Apresentar:
- Tabela de métricas  
- Gráfico de resíduos  
- Predito vs real  

### 5.6. Artefatos
- `model.pkl`  
- `metrics.json`  
- `feature_importance.csv` (se aplicável)  

---

## 6) Ética, governança e uso responsável
- Garantir anonimização de atributos sensíveis  
- Monitorar vieses regionais ou socioeconômicos  
- Usar o modelo apenas como suporte à decisão  
- Documentar limitações e incertezas  

---

## 7) Critérios de aceite (checklist)
- [ ] Pipeline SOR → SOT → SPEC implementado  
- [ ] EDA com gráficos e insights consistentes  
- [ ] Modelo treinado com split temporal  
- [ ] Métricas registradas e interpretadas  
- [ ] Artefatos salvos e reproduzíveis  

---

## 8) Próximos passos (opcional)
- Ajuste de hiperparâmetros  
- Explainability (SHAP)  
- Validação por subgrupos (sexo, faixa etária, especialidade)  