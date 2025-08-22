# Avaliacao_pratica_01837-2025_Mentor_de_Projetos_e_Pesquisa_FIESC
Cargo: Mentor de Projetos e Pesquisa


# Pipeline de Processamento e Modelagem  
### Autor: Bryan Ambrósio  
### Agosto de 2025

Este repositório contém um pipeline completo para conversão, preparação, análise e modelagem de dados de sensores.  
Os scripts devem ser executados **nesta ordem**:  

---

## 1. `1-npy_to_csv.py`  
- Varre a pasta `data_raw/` em busca de arquivos `*.npy`.  
- Converte cada arquivo para `.csv` e salva em `data_csv/`.  
- Mantém o mesmo nome de arquivo, apenas mudando a extensão para `.csv`.  

---

## 2. `2-csv_statistics.py`  
- Lê todos os `.csv` da pasta `data_csv/`.  
- Calcula estatísticas gerais (mínimo, máximo, média, desvio padrão) para todas as colunas numéricas.  
- Gera uma tabela de resumo no terminal, exibindo:  


---

## 3. `3-prepare_datasets.py`  
- Lê os arquivos de sensores (`Dados_1.csv ... Dados_5.csv`) e o arquivo de classes (`Classes.csv`).  
- Realiza limpeza e conversão numérica (corrigindo notação pt-BR, removendo colunas extras, etc.).  
- Salva versões limpas na pasta `data-prepared/`.  
- Junta cada sensor com suas classes em `joined/Joined_k.csv`.  
- Cria um arquivo combinado `Joined_All.csv` com todos os sensores alinhados e valores faltantes imputados pela mediana.  

---

## 4. `4-isolationforest_model.py`  
- Lê `Joined_All.csv` e divide em treino/teste (estratificado por sensor).  
- Normaliza dados por sensor (mediana/IQR).  
- Aplica PCA no treino e transforma treino/teste.  
- Treina um modelo **IsolationForest** para detecção de anomalias.  
- Gera figuras PCA 2D (por classe e por sensor) destacando anomalias.  
- Exporta embeddings + scores em CSV.  
- Salva o pipeline completo em `results_pca_if_split_sbs/pca_if_model.joblib`, pronto para uso posterior em inferência.  

---

### Saídas principais:
- **`data_csv/`** → conversões de `.npy` para `.csv`.  
- **`data-prepared/` e `joined/`** → arquivos tratados e combinados.  
- **`results_pca_if_split_sbs/`** → gráficos, CSVs e modelo final salvo.  

---

## Como executar
Execute cada script sequencialmente:  

```bash
python 1-npy_to_csv.py
python 2-csv_statistics.py
python 3-prepare_datasets.py
python 4-isolationforest_model.py
```

## Apresentação  

O repositório também contém a apresentação **`App_Mentor_FIESC.pptx`**, que deve ser consultada para obter **mais detalhes sobre o projeto**, sua motivação, arquitetura e principais resultados.  
