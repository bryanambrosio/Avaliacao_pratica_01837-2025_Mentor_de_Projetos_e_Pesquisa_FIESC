"""
===============================================================================
 Autor: Bryan Ambrósio
 Data: Agosto de 2025

 O que faz:
 ----------
 - Varre a pasta data_csv/ em busca de *.csv
 - Para cada arquivo, imprime:
     Arquivo | Shape | Dtype | Mínimo | Máximo | Média | Desvio Padrão
 - As estatísticas são calculadas agregando todas as colunas numéricas do CSV.
 - Se não houver coluna numérica, mostra "-".
===============================================================================
"""

from pathlib import Path   # Manipulação de caminhos de arquivos/pastas
import numpy as np         # Operações numéricas e estatísticas
import pandas as pd        # Leitura e manipulação de arquivos CSV

# Caminho para a pasta onde estão os arquivos CSV
CSV_DIR = Path("data_csv")

def fmt_num(x):
    """
    Formata número de forma curta e legível (pt-BR), até 6 casas decimais.
    - Substitui ponto por vírgula como separador decimal.
    - Retorna '-' para valores nulos, NaN ou infinitos.
    """
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "-"
    return f"{x:,.6f}".replace(",", "X").replace(".", ",").replace("X", ".")

def summarize_csv(csv_path: Path):
    """
    Lê um CSV e retorna um resumo com:
    (arquivo, shape, tipo_dados, min, max, mean, std).
    
    - Detecta o tipo predominante de dados (dtype).
    - Calcula estatísticas agregadas considerando TODAS as colunas numéricas.
    """
    # Lê o CSV, usando low_memory=False para evitar inferência incorreta de tipos
    df = pd.read_csv(csv_path, low_memory=False)
    n_rows, n_cols = df.shape  # Obtém número de linhas e colunas

    # Determina a string de tipo de dados predominante no arquivo
    dtypes = df.dtypes
    unique_dtypes = dtypes.astype(str).unique()
    if len(unique_dtypes) == 1:
        # Se todas as colunas têm o mesmo dtype, usa esse tipo
        dtype_str = unique_dtypes[0]
    else:
        # Caso contrário, verifica se maioria é numérica
        num = df.select_dtypes(include=[np.number])
        if num.shape[1] >= max(1, int(0.9 * n_cols)):  # >=90% numérico
            dtype_str = str(num.dtypes.mode()[0])  # Tipo mais frequente
        else:
            dtype_str = "mixed"  # Tipos mistos

    # Seleciona apenas colunas numéricas
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] > 0:  # Se houver colunas numéricas
        vals = num.to_numpy().ravel()  # Flatten para vetor 1D
        if vals.size > 0:
            # Calcula estatísticas ignorando NaN
            mn = np.nanmin(vals)
            mx = np.nanmax(vals)
            mean = np.nanmean(vals)
            std = np.nanstd(vals, ddof=1)  # Desvio padrão amostral
        else:
            mn = mx = mean = std = None
    else:
        # Se não houver colunas numéricas
        mn = mx = mean = std = None

    # Retorna lista com informações formatadas
    return [
        csv_path.name,
        f"({n_rows:,}, {n_cols:,})".replace(",", "."),
        dtype_str,
        fmt_num(mn),
        fmt_num(mx),
        fmt_num(mean),
        fmt_num(std),
    ]

def print_table(rows):
    """
    Imprime a tabela formatada com cabeçalho e alinhamento.
    - Calcula largura de cada coluna para alinhamento uniforme.
    """
    headers = ["Arquivo", "Shape", "Dtype", "Mínimo", "Máximo", "Média", "Desvio Padrão"]
    # Determina a largura de cada coluna
    col_widths = [max(len(h), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    
    # Função interna para imprimir uma linha formatada
    def print_row(cols):
        print("  ".join(str(c).ljust(w) for c, w in zip(cols, col_widths)))
    
    print("\n================= Resumo das características dos arquivos .csv =================")
    print_row(headers)
    print_row(["-" * len(h) for h in headers])  # Linha separadora
    for r in rows:
        print_row(r)

def main():
    """
    Função principal:
    - Lista todos os arquivos .csv na pasta CSV_DIR.
    - Para cada arquivo, chama summarize_csv().
    - Imprime a tabela final com resultados.
    """
    csv_files = sorted(CSV_DIR.glob("*.csv"))  # Lista ordenada dos CSVs
    if not csv_files:
        print(f"[aviso] Nenhum CSV encontrado em: {CSV_DIR.resolve()}")
        return

    rows = []
    for p in csv_files:
        try:
            rows.append(summarize_csv(p))
        except Exception as e:
            # Em caso de erro na leitura/processamento do arquivo
            rows.append([p.name, "-", "-", "-", "-", "-", f"erro: {e}"])

    print_table(rows)

if __name__ == "__main__":
    # Executa a função principal apenas se o script for rodado diretamente
    main()
