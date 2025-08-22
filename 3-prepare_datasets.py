"""
===============================================================================
 Preparação e junção de arquivos de sensores e classes
 Autor: Bryan Ambrósio
 Data: Agosto de 2025

 O que faz:
 ----------
 1. Lê arquivos de dados de sensores (Dados_1.csv ... Dados_5.csv) e o arquivo
    de classes (Classes.csv) a partir da pasta data_csv/.
 2. Aplica regras específicas para cada arquivo (remover última coluna, converter
    notação pt-BR, etc.).
 3. Converte todas as colunas para numérico, tratando valores faltantes e formatos.
 4. Salva versões "preparadas" (limpas) na pasta data-prepared/.
 5. Junta cada sensor com suas classes em arquivos Joined_k.csv na pasta joined/.
 6. Cria um arquivo Joined_All.csv com todos os sensores alinhados e com imputação
    de valores faltantes pela mediana.
===============================================================================
"""

from pathlib import Path
import numpy as np
import pandas as pd

# Diretórios de entrada e saída
DATA_IN = Path("data_csv")                # CSVs brutos
DATA_OUT = Path("data-prepared")          # CSVs preparados
JOIN_DIR = DATA_OUT / "joined"            # CSVs preparados + classes + dataset final (Joined_all.csv)

# Regras específicas para cada arquivo de dados
FILES_RULES = {
    "Dados_1.csv": {"drop_last_col": True,  "force_ptbr": False},
    "Dados_2.csv": {"drop_last_col": True,  "force_ptbr": False},
    "Dados_3.csv": {"drop_last_col": True,  "force_ptbr": False},
    "Dados_4.csv": {"drop_last_col": False, "force_ptbr": False},
    "Dados_5.csv": {"drop_last_col": False, "force_ptbr": True},  # vírgula decimal pt-BR
}
CLASSES_NAME = "Classes.csv"  # Nome do arquivo de classes

# ----------------- Funções auxiliares -----------------

def read_raw_without_header(path: Path) -> pd.DataFrame:
    """
    Lê um CSV detectando automaticamente o separador (sep=None, engine='python'),
    ignora a primeira linha (skiprows=1) e lê todo conteúdo como string (dtype=str).
    Não define cabeçalho (header=None).
    """
    return pd.read_csv(path, sep=None, engine="python", header=None, skiprows=1, dtype=str)

def to_numeric_series(s: pd.Series, force_ptbr: bool = False) -> pd.Series:
    """
    Converte uma série de strings em valores float, tratando:
    - Espaços, caracteres de milhar, apóstrofos e aspas.
    - Substitui vírgula decimal por ponto se for formato pt-BR.
    - 'force_ptbr' força a conversão para formato pt-BR.
    - Valores vazios, '-', 'nan', 'None' viram NaN.
    """
    s = s.astype(str).str.strip()
    s = s.replace({"": np.nan, "-": np.nan, "nan": np.nan, "NaN": np.nan, "None": np.nan, "none": np.nan})
    
    if force_ptbr:
        # Remove espaços e milhar, troca vírgula por ponto
        s = (s.str.replace(" ", "", regex=False)
               .str.replace("’", "", regex=False).str.replace("'", "", regex=False)
               .str.replace(".", "", regex=False)      # milhar
               .str.replace(",", ".", regex=False))    # decimal
        return pd.to_numeric(s, errors="coerce")
    
    # Caso não seja forçado, detecta automaticamente se há vírgula
    has_comma = s.str.contains(",", na=False)
    s_pt = s.where(
        ~has_comma,
        s.str.replace(" ", "", regex=False)
         .str.replace("’", "", regex=False).str.replace("'", "", regex=False)
         .str.replace(".", "", regex=False)   # milhar
         .str.replace(",", ".", regex=False)  # decimal
    )
    return pd.to_numeric(s_pt, errors="coerce")

def prepare_classes() -> pd.Series:
    """
    Prepara o arquivo de classes:
    - Remove a primeira linha (cabeçalho).
    - Salva em data-prepared/Classes.csv.
    - Retorna uma Series com as classes como strings.
    """
    src = DATA_IN / CLASSES_NAME
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        raise FileNotFoundError(f"Não encontrei {src.resolve()}")

    with open(src, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    if not lines:
        raise ValueError(f"{CLASSES_NAME} está vazio.")

    dst = DATA_OUT / CLASSES_NAME
    with open(dst, "w", encoding="utf-8", errors="ignore", newline="") as f:
        f.writelines(lines[1:])  # remove cabeçalho

    y = pd.read_csv(dst, header=None).iloc[:, 0].astype(str).reset_index(drop=True)
    print(f"[OK] {CLASSES_NAME}: {len(y)} linhas (sem cabeçalho) → {dst.as_posix()}")
    return y

def prepare_dados(name: str, rules: dict) -> pd.DataFrame:
    """
    Prepara um arquivo de dados de sensores:
    - Remove a primeira linha.
    - Remove a última coluna, se especificado em 'drop_last_col'.
    - Converte todas as colunas para numérico, aplicando 'force_ptbr' se necessário.
    - Salva o resultado limpo na pasta data-prepared/.
    """
    src = DATA_IN / name
    if not src.exists():
        raise FileNotFoundError(f"Não encontrei {src.resolve()}")

    df = read_raw_without_header(src)
    if df.shape[1] == 0:
        raise ValueError(f"{name}: após remover a primeira linha, não sobrou nenhuma coluna.")

    if rules.get("drop_last_col", False):
        if df.shape[1] < 2:
            raise ValueError(f"{name}: não há como remover a última coluna (apenas {df.shape[1]} coluna).")
        df = df.iloc[:, :-1]

    # Converte cada coluna para numérico
    num = pd.DataFrame({c: to_numeric_series(df[c], force_ptbr=rules.get("force_ptbr", False)) for c in df.columns})

    DATA_OUT.mkdir(parents=True, exist_ok=True)
    out_prep = DATA_OUT / name
    num.to_csv(out_prep, index=False, header=False)
    print(f"[OK] {name}: {num.shape[0]} linhas × {num.shape[1]} colunas → {out_prep.as_posix()}")
    return num

def join_with_classes(k: int, dfX: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Junta dados de um sensor com as classes:
    - Equaliza o número de linhas (trunca para o menor tamanho, se necessário).
    - Renomeia as features para col_1, col_2, ...
    - Adiciona colunas 'classe' e 'sensor_id'.
    - Salva como Joined_k.csv.
    """
    n_x, n_y = len(dfX), len(y)
    if n_x != n_y:
        m = min(n_x, n_y)
        print(f"[WARN] Sensor_{k}: linhas diferentes (X={n_x}, classes={n_y}) → usando as primeiras {m}.")
        dfX = dfX.iloc[:m, :].reset_index(drop=True)
        y = y.iloc[:m].reset_index(drop=True)

    feat_cols = [f"col_{i+1}" for i in range(dfX.shape[1])]
    dfX.columns = feat_cols

    joined = pd.DataFrame({"classe": y, "sensor_id": f"Sensor_{k}"})
    joined = pd.concat([joined, dfX], axis=1)

    JOIN_DIR.mkdir(parents=True, exist_ok=True)
    out_join = JOIN_DIR / f"Joined_{k}.csv"
    joined.to_csv(out_join, index=False)
    print(f"[OK] Joined_{k}: {joined.shape[0]} linhas × {joined.shape[1]} colunas → {out_join.as_posix()}")
    return joined

def impute_median_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preenche valores NaN das features numéricas com a mediana de cada coluna.
    Mantém as colunas 'classe' e 'sensor_id' sem alteração.
    """
    non_feat = ["classe", "sensor_id"]
    feat = df.drop(columns=non_feat, errors="ignore")
    med = feat.median(axis=0, skipna=True, numeric_only=True).fillna(0.0)
    feat_imp = feat.fillna(med)
    return pd.concat([df[["classe", "sensor_id"]], feat_imp], axis=1)

# ----------------- Função principal -----------------

def main():
    # 1) Prepara o arquivo de classes
    y = prepare_classes()

    # 2) Prepara cada arquivo de dados conforme as regras
    prepared = {}
    sensor_map = {}
    for k, name in enumerate(FILES_RULES.keys(), start=1):
        df = prepare_dados(name, FILES_RULES[name])
        prepared[name] = df
        sensor_map[k] = name

    print("\n=== Shapes após preparação (antes dos joins) ===")
    for k, name in sensor_map.items():
        df = prepared[name]
        print(f" - {name}: {df.shape[0]} × {df.shape[1]}")

    # 3) Junta cada sensor com as classes
    joined_list = []
    for k, name in sensor_map.items():
        dfX = prepared[name]
        joined = join_with_classes(k, dfX, y)
        joined_list.append(joined)

    # 4) Alinha todos os joined para ter o mesmo conjunto de colunas (união)
    all_cols = ["classe", "sensor_id"] + sorted(set().union(*[set(df.columns) for df in joined_list]) - {"classe", "sensor_id"})
    aligned = [df.reindex(columns=all_cols) for df in joined_list]

    # 5) Concatena todos os sensores e imputa valores faltantes pela mediana
    big = pd.concat(aligned, axis=0, ignore_index=True)
    big_imp = impute_median_features(big)

    # 6) Salva arquivo final combinado
    out_all = JOIN_DIR / "Joined_All.csv"
    big_imp.to_csv(out_all, index=False)

    print("\n=== Resumo final ===")
    for k, df in enumerate(joined_list, start=1):
        print(f" - Joined_{k}: {df.shape[0]} × {df.shape[1]}")
    print(f" - Joined_All: {big_imp.shape[0]} × {big_imp.shape[1]} → {out_all.as_posix()}")

if __name__ == "__main__":
    main()
