"""
===============================================================================
 Conversão de arquivos .npy para .csv
 Autor: Bryan Ambrósio
 Data: Agosto de 2025

 O que faz:
 ----------
 - Varre a pasta data_raw/ em busca de arquivos *.npy
 - Carrega o conteúdo de cada .npy
 - Salva um arquivo .csv correspondente na pasta data_csv/
 - Mantém o mesmo nome do arquivo, alterando apenas a extensão para .csv
===============================================================================
"""

from pathlib import Path
import numpy as np
import pandas as pd

# Pastas de entrada e saída
RAW_DIR = Path("data_raw")
CSV_DIR = Path("data_csv")
CSV_DIR.mkdir(exist_ok=True)  # Cria a pasta de saída, se não existir

def npy_para_csv(npy_path: Path):
    """
    Lê um arquivo .npy e salva como .csv na pasta de saída.
    """
    try:
        # Carrega o conteúdo do .npy
        data = np.load(npy_path, allow_pickle=True)

        # Se for array 1D, transforma em 2D para salvar em CSV
        if data.ndim == 1:
            df = pd.DataFrame(data, columns=["valor"])
        else:
            df = pd.DataFrame(data)

        # Caminho do CSV de saída
        csv_path = CSV_DIR / (npy_path.stem + ".csv")

        # Salva como CSV
        df.to_csv(csv_path, index=False)
        print(f"[OK] {npy_path.name} -> {csv_path.name}")
    
    except Exception as e:
        print(f"[ERRO] Falha ao converter {npy_path.name}: {e}")

def main():
    """
    Percorre a pasta RAW_DIR e converte todos os .npy encontrados para .csv.
    """
    arquivos_npy = sorted(RAW_DIR.glob("*.npy"))

    if not arquivos_npy:
        print(f"[aviso] Nenhum arquivo .npy encontrado em: {RAW_DIR.resolve()}")
        return

    for npy_file in arquivos_npy:
        npy_para_csv(npy_file)

if __name__ == "__main__":
    main()
