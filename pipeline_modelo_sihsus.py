from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


ROOT = Path(__file__).resolve().parent
SPEC_PATH = ROOT / "spec_sih_2025_sp.parquet"
ARTIFACTS_DIR = ROOT / "artifacts"
OUTPUTS_DIR = ROOT / "outputs"


def gerar_base_sintetica(n: int = 6000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mes = rng.integers(1, 13, size=n)
    idade = rng.integers(0, 95, size=n)
    tempo = rng.integers(1, 25, size=n)
    sexo = rng.choice(["M", "F"], size=n)
    cid = rng.choice(["J18", "I21", "K35", "N39", "A41", "S72"], size=n)
    especialidade = rng.choice(
        ["CLINICA_MEDICA", "CARDIOLOGIA", "ORTOPEDIA", "PEDIATRIA", "CIRURGIA_GERAL"],
        size=n,
    )
    municipio = rng.choice(["SAO_PAULO", "CAMPINAS", "SANTOS", "RIBEIRAO_PRETO", "SOROCABA"], size=n)

    base_custo = (
        1200
        + tempo * 310
        + (idade > 65) * 1200
        + (especialidade == "CARDIOLOGIA") * 1600
        + (especialidade == "ORTOPEDIA") * 950
        + (cid == "A41") * 2400
        + (cid == "I21") * 1800
        + rng.normal(0, 700, size=n)
    )
    valor = np.clip(base_custo, 200, None)

    return pd.DataFrame(
        {
            "mes_internacao": mes,
            "idade": idade,
            "tempo_internacao": tempo,
            "sexo": sexo,
            "cid_principal": cid,
            "especialidade": especialidade,
            "municipio": municipio,
            "valor_total_internacao": valor,
        }
    )


def carregar_dados() -> Tuple[pd.DataFrame, str]:
    if SPEC_PATH.exists():
        df = pd.read_parquet(SPEC_PATH)
        return df, "arquivo_spec"

    df = gerar_base_sintetica()
    df.to_parquet(SPEC_PATH, index=False)
    return df, "base_sintetica"


def normalizar_colunas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.lower() for c in out.columns]
    if "val_tot" in out.columns and "valor_total_internacao" not in out.columns:
        out = out.rename(columns={"val_tot": "valor_total_internacao"})

    if "valor_total_internacao" not in out.columns:
        raise ValueError("A coluna valor_total_internacao não foi encontrada na base.")

    if "mes_internacao" not in out.columns:
        if "dt_inter" in out.columns:
            out["dt_inter"] = pd.to_datetime(out["dt_inter"], errors="coerce")
            out["mes_internacao"] = out["dt_inter"].dt.month
        elif "competencia" in out.columns:
            out["mes_internacao"] = (
                out["competencia"].astype(str).str.slice(4, 6).replace("", np.nan).astype(float)
            )
        else:
            raise ValueError("Não foi possível derivar mes_internacao para o split temporal.")

    out = out[pd.to_numeric(out["valor_total_internacao"], errors="coerce").notna()]
    out["valor_total_internacao"] = out["valor_total_internacao"].astype(float)
    out = out[out["valor_total_internacao"] > 0]
    out = out[pd.to_numeric(out["mes_internacao"], errors="coerce").notna()]
    out["mes_internacao"] = out["mes_internacao"].astype(int)
    out = out[(out["mes_internacao"] >= 1) & (out["mes_internacao"] <= 12)]
    return out


def split_temporal(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    treino = df[df["mes_internacao"].between(1, 9)].copy()
    teste = df[df["mes_internacao"].between(10, 12)].copy()
    if treino.empty or teste.empty:
        raise ValueError("Split temporal inválido: é necessário ter dados de treino (1-9) e teste (10-12).")
    return treino, teste


def construir_modelo(num_cols: list[str], cat_cols: list[str]) -> Pipeline:
    modelo = HistGradientBoostingRegressor(max_depth=8, max_iter=300, learning_rate=0.08, random_state=42)

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
            ),
        ]
    )
    num_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

    preprocessador = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(steps=[("preprocess", preprocessador), ("model", modelo)])


def ajustar_pipeline(modelo: Pipeline, x_treino: pd.DataFrame, y_treino: pd.Series) -> Pipeline:
    modelo.fit(x_treino, np.log1p(y_treino))
    return modelo


def avaliar_modelo(modelo: Pipeline, x_teste: pd.DataFrame, y_teste: pd.Series) -> Tuple[dict, np.ndarray]:
    y_pred_log = modelo.predict(x_teste)
    y_pred = np.expm1(y_pred_log)
    y_pred = np.clip(y_pred, 0, None)

    mae = mean_absolute_error(y_teste, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_teste, y_pred)))
    r2 = r2_score(y_teste, y_pred)
    mape = float(np.mean(np.abs((y_teste - y_pred) / np.maximum(y_teste, 1e-9))) * 100)

    metrics = {
        "MAE": float(mae),
        "RMSE": rmse,
        "R2": float(r2),
        "MAPE": mape,
    }
    return metrics, y_pred


def salvar_graficos(y_real: pd.Series, y_pred: np.ndarray) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    residuos = y_real - y_pred
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuos, alpha=0.25)
    plt.axhline(0, color="red", linestyle="--", linewidth=1)
    plt.title("Resíduos do modelo")
    plt.xlabel("Valor predito")
    plt.ylabel("Resíduo (real - predito)")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "residuos_modelo.png", dpi=120)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(y_real, y_pred, alpha=0.25)
    lim_min = float(min(y_real.min(), y_pred.min()))
    lim_max = float(max(y_real.max(), y_pred.max()))
    plt.plot([lim_min, lim_max], [lim_min, lim_max], color="red", linestyle="--", linewidth=1)
    plt.title("Predito vs Real")
    plt.xlabel("Valor real")
    plt.ylabel("Valor predito")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "predito_vs_real.png", dpi=120)
    plt.close()


def salvar_feature_importance(modelo: Pipeline, x_teste: pd.DataFrame, y_teste: pd.Series) -> pd.DataFrame:
    x_teste_t = modelo.named_steps["preprocess"].transform(x_teste)
    y_teste_log = np.log1p(y_teste)

    imp = permutation_importance(
        estimator=modelo.named_steps["model"],
        X=x_teste_t,
        y=y_teste_log,
        scoring="neg_mean_absolute_error",
        n_repeats=5,
        random_state=42,
    )

    feat_names = modelo.named_steps["preprocess"].get_feature_names_out().tolist()
    fi = (
        pd.DataFrame({"feature": feat_names, "importance": imp.importances_mean})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    fi.to_csv(ARTIFACTS_DIR / "feature_importance.csv", index=False)
    return fi


def treinar_e_salvar() -> dict:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    dados, fonte = carregar_dados()
    dados = normalizar_colunas(dados)
    treino, teste = split_temporal(dados)

    y_treino = treino["valor_total_internacao"]
    y_teste = teste["valor_total_internacao"]
    x_treino = treino.drop(columns=["valor_total_internacao"])
    x_teste = teste.drop(columns=["valor_total_internacao"])

    num_cols = x_treino.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat_cols = [c for c in x_treino.columns if c not in num_cols]
    pipeline = construir_modelo(num_cols=num_cols, cat_cols=cat_cols)
    pipeline = ajustar_pipeline(pipeline, x_treino, y_treino)
    metrics, y_pred = avaliar_modelo(pipeline, x_teste, y_teste)

    joblib.dump(pipeline, ARTIFACTS_DIR / "model.pkl")
    salvar_graficos(y_teste, y_pred)
    fi = salvar_feature_importance(pipeline, x_teste, y_teste)

    payload = {
        "fonte_dados": fonte,
        "n_treino": int(len(treino)),
        "n_teste": int(len(teste)),
        "metricas": metrics,
        "feature_importance_top10": fi.head(10).to_dict(orient="records"),
    }
    with (ARTIFACTS_DIR / "metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)

    return payload


if __name__ == "__main__":
    resultado = treinar_e_salvar()
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
