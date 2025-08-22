"""
===============================================================================
IsolationForest Model
 Autor: Bryan Ambrósio
 Data: Agosto de 2025

 O que faz:
 ----------
 1) Lê o arquivo combinado Joined_All.csv (classe, sensor_id, features).
 2) Faz split TREINO/TESTE estratificado por sensor_id.
 3) Normaliza por sensor (mediana/IQR) usando APENAS estatísticas do TREINO.
 4) Ajusta PCA no TREINO e transforma TREINO/TESTE.
 5) Ajusta IsolationForest no TREINO; gera scores e flags de anomalia.
 6) Plota PCA 2D lado a lado (por classe e por sensor), marcando anomalias.
 7) Exporta embeddings + scores (CSV) para auditoria.
 8) SALVA O MODELO (pipeline de inferência):
      - stats de normalização por sensor (mediana/IQR)
      - listas de colunas/ordem das features
      - objeto PCA
      - objeto IsolationForest
      - parâmetros relevantes (contamination, n_components, etc.)
 9) Mostra exemplo de como carregar e usar o modelo salvo.

 Saídas:
 -------
 - Pasta results_pca_if_split_sbs/ com PNGs e CSVs.
 - Arquivo results_pca_if_split_sbs/pca_if_model.joblib com o pipeline salvo.
===============================================================================
"""

import os
os.environ["MPLBACKEND"] = "Agg"  # backend headless para rodar sem GUI

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()  # garante modo não interativo

from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import joblib  # >>> para salvar/carregar o modelo

# ------------------------ Parâmetros gerais ------------------------
IN_PATH = Path("data-prepared/joined/Joined_All.csv")
OUT_DIR = Path("results_pca_if_split_sbs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_COMPONENTS = 10               # desejado (limitado mais adiante por n e p)
SUBSAMPLE_FOR_PLOTS = 50000     # subamostragem para visual (rapidez)
CONTAMINATION = 0.10            # fração alvo de anomalias por sensor
EPS = 1e-9                      # evita divisão por zero
MODEL_PATH = OUT_DIR / "pca_if_model.joblib"  # <<< caminho do modelo salvo

# ------------------------ Funções utilitárias ------------------------
def choose_ncomp(n, p, wanted):
    """Escolhe n_components válido: no mínimo 2, no máximo min(wanted, p, n-1)."""
    return int(max(2, min(wanted, p, n - 1)))

# ---------- Normalização robusta por sensor (stats do TREINO) ----------
def compute_sensor_stats(X_train: pd.DataFrame, sensor_train: pd.Series):
    """
    Calcula, para cada sensor, mediana e IQR (Q3-Q1) por coluna no conjunto de TREINO.
    - 'scale' usa IQR; onde IQR=0, cai para NaN e depois substituímos por 1.0.
    - Também fornece um fallback global (_GLOBAL_) caso um sensor novo apareça na inferência.
    Retorna: dict[sensor_id] -> {"median": Series, "scale": Series}
    """
    stats = {}
    for s_id, idx in sensor_train.groupby(sensor_train).groups.items():
        block = X_train.loc[idx]
        med = block.median(skipna=True, numeric_only=True)
        q1  = block.quantile(0.25, numeric_only=True)
        q3  = block.quantile(0.75, numeric_only=True)
        iqr = (q3 - q1).replace(0, np.nan)
        scale = iqr.fillna(1.0)
        stats[str(s_id)] = {"median": med, "scale": scale}

    # fallback global (caso um sensor_id não exista nos stats em produção)
    med_g = X_train.median(skipna=True, numeric_only=True)
    iqr_g = (X_train.quantile(0.75) - X_train.quantile(0.25)).replace(0, np.nan).fillna(1.0)
    stats["_GLOBAL_"] = {"median": med_g, "scale": iqr_g}
    return stats

def apply_sensor_stats(X: pd.DataFrame, sensor: pd.Series, stats: dict) -> pd.DataFrame:
    """
    Aplica a normalização robusta por sensor:
    X_norm = (X - mediana_sensor) / (IQR_sensor + EPS)
    - Se o sensor não estiver em 'stats', usa _GLOBAL_.
    - Garante saída float e preenche NaN restantes com 0.0.
    """
    Xn = pd.DataFrame(index=X.index, columns=X.columns, dtype=float)
    for s_id, idx in sensor.groupby(sensor).groups.items():
        key = str(s_id) if str(s_id) in stats else "_GLOBAL_"
        med = stats[key]["median"]
        scale = stats[key]["scale"]
        Xn.loc[idx] = (X.loc[idx] - med) / (scale + EPS)
    return Xn.fillna(0.0)

# ---------- Cores consistentes entre painéis ----------
def make_color_map(labels):
    """
    Cria um mapa de cores estável por rótulo usando 'tab20'.
    """
    uniq = sorted(pd.unique(labels))
    cmap = plt.get_cmap("tab20")
    colors = {u: cmap(i % cmap.N) for i, u in enumerate(uniq)}
    return colors

# ---------- Plot utilitário ----------
def plot_panel(ax, Z2, labels: pd.Series, anom_flags: np.ndarray, title: str, colors_map: dict, legend_max=14):
    """
    Plota dispersão 2D (PC1, PC2) colorida pelo rótulo (classe ou sensor).
    Anomalias (flags True/1) são circuladas em preto.
    """
    uniq = sorted(pd.unique(labels))
    show = uniq[:legend_max]
    for u in uniq:
        m = (labels.values == u)
        ax.scatter(Z2[m,0], Z2[m,1], s=8, alpha=0.7,
                   color=colors_map.get(u, None),
                   label=str(u) if u in show else None)
    if np.any(anom_flags):
        m = anom_flags.astype(bool)
        ax.scatter(Z2[m,0], Z2[m,1], s=30, facecolors="none",
                   edgecolors="black", linewidths=1.0, zorder=3)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    if len(uniq) <= legend_max:
        ax.legend(markerscale=2, fontsize=8, loc="best")

def side_by_side_figure(Z2_tr, Z2_te,
                        labels_tr: pd.Series, labels_te: pd.Series,
                        flags_tr: np.ndarray, flags_te: np.ndarray,
                        title_left: str, title_right: str,
                        out_png: Path, legend_max=14):
    """
    Gera figura com dois painéis (TREINO/TESTE) lado a lado,
    usando o mesmo range em x/y e cores consistentes entre conjuntos.
    """
    colors = make_color_map(pd.concat([labels_tr, labels_te], axis=0))

    xy = np.vstack([Z2_tr, Z2_te])
    x_min, x_max = xy[:,0].min(), xy[:,0].max()
    y_min, y_max = xy[:,1].min(), xy[:,1].max()
    pad_x = 0.05 * (x_max - x_min + 1e-9)
    pad_y = 0.05 * (y_max - y_min + 1e-9)
    xlim = (x_min - pad_x, x_max + pad_x)
    ylim = (y_min - pad_y, y_max + pad_y)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), constrained_layout=True)
    plot_panel(axes[0], Z2_tr, labels_tr, flags_tr, title_left,  colors, legend_max)
    plot_panel(axes[1], Z2_te, labels_te, flags_te, title_right, colors, legend_max)

    for ax in axes:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

# ------------------------ Pipeline principal ------------------------
def main():
    # --- Leitura e checagens iniciais ---
    assert IN_PATH.exists(), f"Arquivo não encontrado: {IN_PATH.resolve()}"
    df = pd.read_csv(IN_PATH)
    assert {"classe","sensor_id"}.issubset(df.columns), "Joined_All.csv precisa de 'classe' e 'sensor_id'."

    # Separa metadados e features
    meta = df[["classe","sensor_id"]].copy()
    X = df.drop(columns=["classe","sensor_id"]).copy()

    # Garante numérico e limpa NaN residual por mediana de coluna (fallback 0.0)
    for c in X.columns:
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    if X.isna().any().any():
        X = X.fillna(X.median(axis=0, skipna=True, numeric_only=True).fillna(0.0))

    # --- Split estratificado por sensor (mantém distribuição de sensores) ---
    idx_all = np.arange(len(X))
    idx_tr, idx_te = train_test_split(
        idx_all, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=meta["sensor_id"]
    )
    X_tr, X_te = X.iloc[idx_tr].reset_index(drop=True), X.iloc[idx_te].reset_index(drop=True)
    meta_tr, meta_te = meta.iloc[idx_tr].reset_index(drop=True), meta.iloc[idx_te].reset_index(drop=True)

    # --- Normalização robusta por sensor: calcula stats no TREINO e aplica ---
    stats = compute_sensor_stats(X_tr, meta_tr["sensor_id"])
    Xn_tr = apply_sensor_stats(X_tr, meta_tr["sensor_id"], stats)
    Xn_te = apply_sensor_stats(X_te, meta_te["sensor_id"], stats)

    # --- PCA (fit no TREINO) ---
    ncomp = choose_ncomp(len(Xn_tr), Xn_tr.shape[1], N_COMPONENTS)
    pca = PCA(n_components=ncomp, random_state=RANDOM_STATE, svd_solver="randomized")
    Z_tr = pca.fit_transform(Xn_tr.values.astype(np.float32))
    Z_te = pca.transform(Xn_te.values.astype(np.float32))
    Z2_tr, Z2_te = Z_tr[:, :2], Z_te[:, :2]  # para visual

    # --- IsolationForest (fit no TREINO) ---
    iforest = IsolationForest(
        n_estimators=500,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ).fit(Z_tr)

    # Scores (maior = mais normal)
    sc_tr = iforest.decision_function(Z_tr)
    sc_te = iforest.decision_function(Z_te)

    # --- Limiar por sensor (definido no TREINO) ---
    thr_by_sensor = {}
    flags_tr = np.zeros(len(sc_tr), dtype=int)
    flags_te = np.zeros(len(sc_te), dtype=int)

    for s in meta_tr["sensor_id"].unique():
        m_tr = (meta_tr["sensor_id"].values == s)
        thr = np.quantile(sc_tr[m_tr], CONTAMINATION)  # quantil por sensor
        thr_by_sensor[s] = float(thr)
        flags_tr[m_tr] = (sc_tr[m_tr] <= thr).astype(int)

    for s in meta_te["sensor_id"].unique():
        thr = thr_by_sensor.get(s, np.quantile(sc_tr, CONTAMINATION))  # fallback global
        m_te = (meta_te["sensor_id"].values == s)
        flags_te[m_te] = (sc_te[m_te] <= thr).astype(int)

    # --- Subamostragem para plots (performance) ---
    rng = np.random.default_rng(RANDOM_STATE)
    nplot_tr = min(SUBSAMPLE_FOR_PLOTS, len(Z2_tr))
    nplot_te = min(SUBSAMPLE_FOR_PLOTS, len(Z2_te))
    idxp_tr = rng.choice(np.arange(len(Z2_tr)), size=nplot_tr, replace=False)
    idxp_te = rng.choice(np.arange(len(Z2_te)), size=nplot_te, replace=False)

    # --- Figuras lado a lado ---
    # Por CLASSE
    side_by_side_figure(
        Z2_tr[idxp_tr], Z2_te[idxp_te],
        meta_tr["classe"].iloc[idxp_tr], meta_te["classe"].iloc[idxp_te],
        flags_tr[idxp_tr].astype(bool),   flags_te[idxp_te].astype(bool),
        title_left ="PCA 2D — Classe (TREINO)  círculos=anomalias",
        title_right="PCA 2D — Classe (TESTE)   círculos=anomalias",
        out_png=OUT_DIR / "pca_by_class_with_if_train_test.png",
        legend_max=14
    )
    # Por SENSOR
    side_by_side_figure(
        Z2_tr[idxp_tr], Z2_te[idxp_te],
        meta_tr["sensor_id"].iloc[idxp_tr], meta_te["sensor_id"].iloc[idxp_te],
        flags_tr[idxp_tr].astype(bool),      flags_te[idxp_te].astype(bool),
        title_left ="PCA 2D — Sensor (TREINO)  círculos=anomalias",
        title_right="PCA 2D — Sensor (TESTE)   círculos=anomalias",
        out_png=OUT_DIR / "pca_by_sensor_with_if_train_test.png",
        legend_max=14
    )

    # --- CSVs opcionais para auditoria (embeddings + scores) ---
    def save_embed(set_name, meta_df, Z, scores, flags):
        out = pd.DataFrame({
            "set": set_name,
            "classe": meta_df["classe"],
            "sensor_id": meta_df["sensor_id"],
            "iforest_score": scores,
            "anomaly_flag": flags.astype(int),
        })
        for i in range(ncomp):
            out[f"PC{i+1}"] = Z[:, i]
        out.to_csv(OUT_DIR / f"pca_if_embeddings_{set_name}_pca{ncomp}.csv", index=False)

    save_embed("train", meta_tr, Z_tr, sc_tr, flags_tr)
    save_embed("test",  meta_te, Z_te, sc_te, flags_te)

    # -------------------- SALVAR O MODELO (PIPELINE) --------------------
    # Empacotamos tudo o que a API precisa para reproduzir a inferência:
    # - order das colunas de entrada (features) esperadas
    # - stats de normalização por sensor (TREINO) + fallback global
    # - objeto PCA ajustado
    # - objeto IsolationForest ajustado
    # - thresholds por sensor (opcional: se quiser usar o mesmo corte da fase de treino)
    # - parâmetros de controle
    model_bundle = {
        "version": "1.0",
        "random_state": RANDOM_STATE,
        "feature_columns": list(X.columns),   # ordem esperada das features NA ENTRADA
        "sensor_stats": stats,               # median/iqr por sensor + _GLOBAL_
        "pca": pca,
        "n_components": ncomp,
        "iforest": iforest,
        "contamination": CONTAMINATION,
        "threshold_by_sensor": thr_by_sensor,  # corte por sensor usado no treino
        "eps": EPS,
    }
    joblib.dump(model_bundle, MODEL_PATH)

    # --- Logs rápidos ---
    print("Figuras (lado a lado) em:", OUT_DIR.resolve())
    print(" - pca_by_class_with_if_train_test.png")
    print(" - pca_by_sensor_with_if_train_test.png")
    print("CSV de embeddings/scores em:", OUT_DIR.resolve())
    print(" - pca_if_embeddings_train_pca{}.csv".format(ncomp))
    print(" - pca_if_embeddings_test_pca{}.csv".format(ncomp))
    print("Modelo salvo em:", MODEL_PATH.resolve())

    print("\nFração de anomalias por sensor (treino):")
    for s in meta_tr["sensor_id"].unique():
        m = (meta_tr["sensor_id"] == s).values
        print(f"   {s}: {flags_tr[m].mean():.3f}")
    print("Fração de anomalias por sensor (teste):")
    for s in meta_te["sensor_id"].unique():
        m = (meta_te["sensor_id"] == s).values
        print(f"   {s}: {flags_te[m].mean():.3f}")

# ------------------------ Exemplo de uso na API ------------------------
def example_api_inference_usage():
    """
    EXEMPLO de como a API deve carregar e usar o modelo salvo (pseudo-uso):
    - Suponha que chegue um payload com 'sensor_id' e um vetor/linha de features
      com os MESMOS nomes/ordem de 'feature_columns' salvos.
    """
    # Carrega o bundle
    bundle = joblib.load(MODEL_PATH)
    feature_columns = bundle["feature_columns"]
    stats = bundle["sensor_stats"]
    pca = bundle["pca"]
    iforest = bundle["iforest"]
    thr_by_sensor = bundle["threshold_by_sensor"]
    eps = bundle["eps"]

    # --- Exemplo de payload (linha única) ---
    sensor_id = "Sensor_3"
    x_dict = {col: 0.0 for col in feature_columns}  # aqui viriam os valores reais
    x_df = pd.DataFrame([x_dict])  # DataFrame com uma linha

    # 1) Garantir ordem/colunas esperadas
    x_df = x_df.reindex(columns=feature_columns)
    for c in x_df.columns:
        if not np.issubdtype(x_df[c].dtype, np.number):
            x_df[c] = pd.to_numeric(x_df[c], errors="coerce")
    if x_df.isna().any().any():
        x_df = x_df.fillna(0.0)  # fallback simples; em prod, alinhar à sua política

    # 2) Normalização por sensor
    key = sensor_id if sensor_id in stats else "_GLOBAL_"
    med = stats[key]["median"]
    scale = stats[key]["scale"]
    x_norm = (x_df - med) / (scale + eps)
    x_norm = x_norm.fillna(0.0)

    # 3) PCA + IsolationForest
    z = pca.transform(x_norm.values.astype(np.float32))
    score = iforest.decision_function(z)[0]  # maior = mais normal

    # 4) Classificação de anomalia via threshold do sensor (se disponível)
    thr = thr_by_sensor.get(sensor_id, np.quantile(joblib.load(MODEL_PATH)["iforest"].decision_function(z), CONTAMINATION))
    is_anomaly = int(score <= thr)

    return {
        "sensor_id": sensor_id,
        "iforest_score": float(score),
        "threshold_used": float(thr),
        "anomaly_flag": is_anomaly,
    }

# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
