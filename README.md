# Limpeza e Tratamento de Valores Ausentes Para Análise de Dados

## Introdução

A análise de dados eficaz começa com a limpeza e o tratamento adequados dos valores ausentes. Este projeto demonstra como lidar com valores ausentes em um conjunto de dados utilizando Python, destacando técnicas e práticas recomendadas para assegurar a integridade e a qualidade dos dados.

## Pacotes Python Usados no Projeto


import math
import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import warnings
warnings.filterwarnings('ignore')


## Carregando os Dados

Carregamos o conjunto de dados e definimos uma lista de valores ausentes:


# Lista de valores ausentes
lista_labels_valores_ausentes = ["n/a", "na", "undefined"]

# Carrega o dataset
dataset = pd.read_csv("dataset.csv", na_values=lista_labels_valores_ausentes)

# Shape do dataset
print(dataset.shape)

# Amostra de dados
dataset.head()


## Carregando o Dicionário de Dados


# Carregando o dicionário de dados
dicionario = pd.read_excel("dicionario.xlsx")

# Shape do dicionário
print(dicionario.shape)

# Amostra de dados
dicionario.head(10)


## Análise Exploratória


# Informações sobre o dataset
dataset.info()

# Estatísticas descritivas
dataset.describe()


## Comparação das Colunas com o Dicionário de Dados


# Comparação das colunas entre o dataset e o dicionário de dados
df_compara_colunas = pd.concat([pd.Series(dataset.columns.tolist()), dicionario['Fields']], axis=1)
df_compara_colunas.columns = ['Coluna no Dataset', 'Coluna no Dicionário']
df_compara_colunas


## Renomeando Colunas


# Renomeia colunas
dataset.rename(columns = {'Dur. (ms)': 'Dur (s)', 'Dur. (ms).1': 'Dur (ms)', 'Start ms': 'Start Offset (ms)', 'End ms': 'End Offset (ms)'}, inplace = True)
print(dataset_dsa.columns.tolist())


## Tratamento de Valores Ausentes

### 1. Identificando Valores Ausentes


# Função que calcula o percentual de valores ausentes
def func_dsa_calc_percent_valores_ausentes(df):
    totalCells = np.product(df.shape)
    missingCount = df.isnull().sum()
    totalMissing = missingCount.sum()
    print("O dataset tem", round(((totalMissing/totalCells) * 100), 2), "%", "de valores ausentes.")

func_dsa_calc_percent_valores_ausentes(dataset)

# Função para calcular valores ausentes por coluna
def func_dsa_calc_percent_valores_ausentes_coluna(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * mis_val / len(df)
    mis_val_dtype = df.dtypes
    mis_val_table = pd.concat([mis_val, mis_val_percent, mis_val_dtype], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'Valores Ausentes', 1 : '% de Valores Ausentes', 2: 'Dtype'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:,0] != 0].sort_values('% de Valores Ausentes', ascending = False).round(2)
    print ("O dataset tem " + str(df.shape[1]) + " colunas.\n" "Encontrado: " + str(mis_val_table_ren_columns.shape[0]) + " colunas que têm valores ausentes.")
    if mis_val_table_ren_columns.shape[0] == 0:
        return
    return mis_val_table_ren_columns

# Cria tabela com valores ausentes
df_missing = func_dsa_calc_percent_valores_ausentes_coluna(dataset_dsa)
df_missing


### 2. Drop de Colunas


# Colunas que serão removidas
colunas_para_remover = df_missing[df_missing['% de Valores Ausentes'] >= 30.00].index.tolist()
colunas_para_remover = [col for col in colunas_para_remover if col not in ['TCP UL Retrans. Vol (Bytes)', 'TCP DL Retrans. Vol (Bytes)']]

# Drop das colunas e cria outro dataframe
dataset_limpo = dataset.drop(colunas_para_remover, axis=1)
print(dataset_limpo.shape)


### 3. Imputação com Preenchimento Reverso


# Imputação de valores ausentes usando backward fill
def func_fix_missing_bfill(df, col):
    count = df[col].isna().sum()
    df[col] = df[col].fillna(method='bfill')
    print(f"{count} valores ausentes na coluna {col} foram substituídos usando o método de preenchimento reverso.")

func_fix_missing_bfill(dataset_dsa_limpo, 'TCP UL Retrans. Vol (Bytes)')
func_fix_missing_bfill(dataset_dsa_limpo, 'TCP DL Retrans. Vol (Bytes)')


### 4. Imputação com Preenchimento Progressivo


# Imputação de valores ausentes usando forward fill
def func_fix_missing_ffill(df, col):
    count = df[col].isna().sum()
    df[col] = df[col].fillna(method='ffill')
    print(f"{count} valores ausentes na coluna {col} foram substituídos usando o método de preenchimento progressivo.")

func_fix_missing_ffill(dataset_dsa_limpo, 'Avg RTT DL (ms)')
func_fix_missing_ffill(dataset_dsa_limpo, 'Avg RTT UL (ms)')


### 5. Imputação de Variáveis Categóricas


# Preenche valor NA para variáveis categóricas
def func_fix_missing_value(df, col, value):
    count = df[col].isna().sum()
    df[col] = df[col].fillna(value)
    if type(value) == 'str':
        print(f"{count} valores ausentes na coluna {col} foram substituídos por '{value}'.")
    else:
        print(f"{count} valores ausentes na coluna {col} foram substituídos por {value}.")

func_fix_missing_value(dataset_dsa_limpo, 'Handset Type', 'unknown')
func_fix_missing_value(dataset_dsa_limpo, 'Handset Manufacturer', 'unknown')


### 6. Drop de Linhas


# Drop de linhas com valores ausentes
def func_drop_linhas_com_na(df):
    old = df.shape[0]
    df.dropna(inplace=True)
    new = df.shape[0]
    count = old - new
    print(f"{count} linhas contendo valores ausentes foram descartadas.")

func_drop_linhas_com_na(dataset_dsa_limpo)


## Conversão de Tipos de Dados


# Converte colunas para datetime
def func_convert_to_datetime(df, columns):
    for col in columns:
        df[col] = pd.to_datetime(df[col])

func_convert_to_datetime(dataset_limpo, ['Start', 'End'])

# Converte colunas para string
def func_dsa_convert_to_string(df, columns):
    for col in columns:
        df[col] = df[col].astype("string")

string_columns = dataset_dsa_limpo.select_dtypes(include='object').columns.tolist()
func_convert_to_string(dataset_limpo, string_columns)

# Converte colunas para int
def func_dsa_convert_to_int(df, columns):
    for col in columns:
        df[col] = df[col].astype("int64")

int_cols = ['Bearer Id', 'IMSI', 'MSISDN/Number', 'IMEI']
func_convert_to_int(dataset_dsa_limpo, int_cols)


## Tratamento de Outliers

# Define a classe TrataOutlier
class TrataOutlier:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def count_outliers(self, Q1, Q3, IQR, columns):
        cut_off = IQR * 1.5
        temp_df = (self.df[columns] < (Q1 - cut_off)) | (self.df[columns] > (Q3 + cut_off))
        return [len(temp_df[temp_df[col] == True]) for col in temp_df]

    def calc_skew(self, columns=None):
        if columns == None:
            columns = self.df.columns
        return [self.df[col].skew() for col in columns]

    def percentage(self, list):
        return [str(round(((value/146887) * 100), 2)) + '%' for value in list]

    def remove_outliers(self, columns):
        for col in columns:
            Q1, Q3 = self.df[col].quantile(0.25), self.df[col].

quantile(0.75)
            IQR = Q3 - Q1
            cut_off = IQR * 1.5
            lower, upper = Q1 - cut_off, Q3 + cut_off
            self.df = self.df.drop(self.df[self.df[col] > upper].index)
            self.df = self.df.drop(self.df[self.df[col] < lower].index)

    def replace_outliers_with_fences(self, columns):
        for col in columns:
            Q1, Q3 = self.df[col].quantile(0.25), self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            cut_off = IQR * 1.5
            lower, upper = Q1 - cut_off, Q3 + cut_off
            self.df[col] = np.where(self.df[col] > upper, upper, self.df[col])
            self.df[col] = np.where(self.df[col] < lower, lower, self.df[col])

    def getOverview(self, columns) -> None:
        min = self.df[columns].min()
        Q1 = self.df[columns].quantile(0.25)
        median = self.df[columns].quantile(0.5)
        Q3 = self.df[columns].quantile(0.75)
        max = self.df[columns].max()
        IQR = Q3 - Q1
        skew = self.calc_skew(columns)
        outliers = self.count_outliers(Q1, Q3, IQR, columns)
        cut_off = IQR * 1.5
        lower, upper = Q1 - cut_off, Q3 + cut_off
        new_columns = ['Nome de Coluna', 'Min', 'Q1', 'Median', 'Q3', 'Max', 'IQR', 'Lower fence', 'Upper fence', 'Skew', 'Num_Outliers', 'Percent_Outliers']
        data = zip([column for column in self.df[columns]], min, Q1, median, Q3, max, IQR, lower, upper, skew, outliers, self.percentage(outliers))
        new_df = pd.DataFrame(data = data, columns = new_columns)
        new_df.set_index('Nome de Coluna', inplace = True)
        return new_df.sort_values('Num_Outliers', ascending=False).transpose()

# Cria o objeto trata outlier
trata_outlier = TrataOutlier(dataset_limpo)

# Lista de colunas float64
lista_colunas = dataset_limpo.select_dtypes('float64').columns.tolist()

# Visão geral dos outliers
trata_outlier.getOverview(lista_colunas)

# Replace dos outliers
trata_outlier.replace_outliers_with_fences(lista_colunas)

# Visão geral dos outliers
trata_outlier.getOverview(lista_colunas)


## Entregando o Resultado da Análise aos Tomadores de Decisão

# Soma dos volumes de dados de upload e download para cada aplicativo
dataset_dsa_limpo['Social Media Data Volume (Bytes)'] = dataset_limpo['Social Media UL (Bytes)'] + dataset_dsa_limpo['Social Media DL (Bytes)']
dataset_dsa_limpo['Google Data Volume (Bytes)'] = dataset_limpo['Google UL (Bytes)'] + dataset_dsa_limpo['Google DL (Bytes)']
dataset_dsa_limpo['Email Data Volume (Bytes)'] = dataset_limpo['Email UL (Bytes)'] + dataset_dsa_limpo['Email DL (Bytes)']
dataset_dsa_limpo['Youtube Data Volume (Bytes)'] = dataset_limpo['Youtube UL (Bytes)'] + dataset_dsa_limpo['Youtube DL (Bytes)']
dataset_dsa_limpo['Netflix Data Volume (Bytes)'] = dataset_limpo['Netflix UL (Bytes)'] + dataset_dsa_limpo['Netflix DL (Bytes)']
dataset_dsa_limpo['Gaming Data Volume (Bytes)'] = dataset_limpo['Gaming UL (Bytes)'] + dataset_dsa_limpo['Gaming DL (Bytes)']
dataset_dsa_limpo['Other Data Volume (Bytes)'] = dataset_limpo['Other UL (Bytes)'] + dataset_dsa_limpo['Other DL (Bytes)']
dataset_dsa_limpo['Total Data Volume (Bytes)'] = dataset_limpo['Total UL (Bytes)'] + dataset_dsa_limpo['Total DL (Bytes)']

# Informações sobre o dataset após a limpeza
dataset_limpo.info()
print(dataset_limpo.shape)
dataset_limpo.head()


## Salvando os Dados Após a Limpeza


# Salvando os dados
dataset_limpo.to_csv('dataset_limpo.csv')


## Conclusão

Este projeto mostrou como lidar com valores ausentes em um conjunto de dados utilizando Python. Foram aplicadas técnicas como drop de colunas, imputação com preenchimento reverso e progressivo, e tratamento de outliers. A limpeza de dados é um passo crucial para garantir a qualidade das análises subsequentes.

### Pontos Importantes

1. **Introdução**: Descrição clara do objetivo do projeto.
2. **Pacotes Python Usados**: Lista de pacotes e imports necessários.
3. **Carregando os Dados**: Passos para carregar e visualizar os dados.
4. **Análise Exploratória**: Informações e estatísticas descritivas dos dados.
5. **Tratamento de Valores Ausentes**: Detalhamento das técnicas usadas para tratar valores ausentes.
6. **Conversão de Tipos de Dados**: Passos para converter os tipos de dados.
7. **Tratamento de Outliers**: Métodos para identificar e tratar outliers.
8. **Entregando o Resultado da Análise**: Processamento adicional para criar novas colunas e sumarizar os dados.
9. **Salvando os Dados**: Passos para salvar os dados limpos.
10. **Conclusão**: Resumo das etapas e importância do tratamento de dados.
