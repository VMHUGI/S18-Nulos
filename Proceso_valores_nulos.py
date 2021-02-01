# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 10:41:05 2021

@author: luigg
"""


import pandas as pd
import os 
import missingno

path='C:/Users/luigg/OneDrive/Documentos/Proyecto_prueba'
os.chdir(path)
df=pd.read_csv('melb_data.csv')


def ordenamiento_nulidad(df, sort=None, axis='columns'):
    """
    Ordena un DataFrame según su nulidad, ya sea en orden ascendente o descendente.
    : param df: el objeto DataFrame que se está ordenando.
    : param sort: el método de clasificación: "ascendente", "descendente" o Ninguno (predeterminado).
    : return: El DataFrame ordenado por nulidad.
    """
    import numpy as np
    if sort is None:
        return df
    elif sort not in ['ascending', 'descending']:
        raise ValueError('The "sort" parameter must be set to "ascending" or "descending".')

    if axis not in ['rows', 'columns']:
        raise ValueError('The "axis" parameter must be set to "rows" or "columns".')

    if axis == 'columns':
        if sort == 'ascending':
            return df.iloc[np.argsort(df.count(axis='columns').values), :]
        elif sort == 'descending':
            return df.iloc[np.flipud(np.argsort(df.count(axis='columns').values)), :]
    elif axis == 'rows':
        if sort == 'ascending':
            return df.iloc[:, np.argsort(df.count(axis='rows').values)]
        elif sort == 'descending':
            return df.iloc[:, np.flipud(np.argsort(df.count(axis='rows').values))]


def filtro_nulidad(df, filter='top', p=0, n=0,exeption=[]):
    """
    Filtra un DataFrame de acuerdo con su nulidad, utilizando alguna combinación de valores numéricos y
    valores porcentuales. Los porcentajes y los umbrales numéricos se pueden especificar simultáneamente: por ejemplo,
    para obtener un DataFrame con columnas de al menos 75% de completitud pero con no más de 5 columnas, use
    `filtro_nulidad (gl, filtro = 'arriba', p = .75, n = 5)`.
    : param df: El DataFrame cuyas columnas se están filtrando.
    : param filter: la orientación del filtro que se aplica al DataFrame. Uno de, "arriba", "abajo",
    o Ninguno (predeterminado). El filtro simplemente devolverá el DataFrame si deja el argumento del filtro sin especificar o
    como Ninguno.
    : param p: un valor de corte del índice de completitud. Si no es cero, el filtro limitará el DataFrame a columnas con al menos p
    lo completo. La entrada debe estar en el rango [0, 1].
    : param n: un valor de corte numérico. Si no es cero, no se devolverá más de este número de columnas.
    : return: El `DataFrame` filtrado por nulidad.
    """
    import numpy as np
    if isinstance(df, pd.DataFrame):
        tmp=pd.DataFrame(df.loc[:, exeption])
        if filter == 'top':
            if p:
                temp=list(set(list(tmp.columns)).intersection(list(df.columns)))
                df = df.iloc[:, [c >= p for c in df.count(axis='rows').values / len(df)]]
                df=df.join(tmp[temp])
            if n:
                df = df.iloc[:, np.sort(np.argsort(df.count(axis='rows').values)[-n:])]
        elif filter == 'bottom':
            if p:
                df = df.iloc[:, [c <= p for c in df.count(axis='rows').values / len(df)]]
            if n:
                df = df.iloc[:, np.sort(np.argsort(df.count(axis='rows').values)[:n])]
                df=df.join(tmp.columns.isin(~df.columns))
        return df

def Reporte_Nulos(df):
  """
    Genera un reporte de un DataFrame de acuerdo con su nulidad, genera dos tablas
    el porcentaje de nulos por variables y correlacion de errores
  """
  if isinstance(df, pd.DataFrame):
      tmp1=pd.DataFrame(df.isnull().sum()).reset_index()
      tmp1.columns=['Variables','Nulos']
      tmp1['Registros']=len(df)
      tmp1['Porcentaje_nulos']=tmp1['Nulos']/tmp1['Registros']
      tmp2 =  df.iloc[:,[i for i, n in enumerate(np.var(df.isnull(), axis='rows')) if n > 0]].isnull().corr()
  else:
      raise Exception('El objeto ingresado no es un DataFrame')
  return tmp1,tmp2

Reporte_Nulos(df)
filtro_nulidad(df,p=0.9,exeption=['BuildingArea'])


group=['dataframe']
def imputacion_basico(messy_df, metric, colnames,group=None):
    """
Se carga una imputacion de datos a los columnas sl
    """
    import numpy as np
    clean_df = messy_df.copy()    
    missing_list = []
    if (isinstance(group, list) & isinstance(colnames, list) ):
  
        if metric=="mean":
            for col in colnames:
                imputenum = messy_df.groupby(group)[col].transform('mean')
                missing_count = messy_df[col].isnull().sum()
                missing_list.append([imputenum]*missing_count)
                clean_df[col] = messy_df[col].fillna(imputenum)            
    
        if metric=="median":
            for col in colnames:
                imputenum = messy_df.groupby(group)[col].transform('median')
                missing_count = messy_df[col].isnull().sum()  
                missing_list.append([imputenum]*missing_count)
                clean_df[col] = messy_df[col].fillna(imputenum)
        
        if metric=="mode":
            for col in colnames:
                imputenum = messy_df.groupby(group)[col].transform('mode')
                missing_count = messy_df[col].isnull().sum()
                missing_pos = clean_df[col].isnull()
                clean_df.loc[clean_df[col].isnull(),col] = np.random.choice(imputenum, missing_count)
                missing_list.append(clean_df.loc[missing_pos,col].tolist())    
    elif  isinstance(colnames, list) :
        print('no group variable detected')
        if metric=="mean":
            for col in colnames:
                imputenum = messy_df[col].mean()
                missing_count = messy_df[col].isnull().sum()
                missing_list.append([imputenum]*missing_count)
                clean_df[col] = messy_df[col].fillna(imputenum)            
    
        if metric=="median":
            for col in colnames:
                imputenum = messy_df[col].median()
                missing_count = messy_df[col].isnull().sum()  
                missing_list.append([imputenum]*missing_count)
                clean_df[col] = messy_df[col].fillna(imputenum)
        
        if metric=="mode":
            for col in colnames:
                imputenum = messy_df[col].mode()
                missing_count = messy_df[col].isnull().sum()
                missing_pos = clean_df[col].isnull()
                clean_df.loc[clean_df[col].isnull(),col] = np.random.choice(imputenum, missing_count)
                missing_list.append(clean_df.loc[missing_pos,col].tolist())
     
    else:
      raise Exception('El objeto ingresado no es una lista')    
    return clean_df, missing_list

clean_df, missing_list=imputacion_basico(df, metric='mean', colnames=['Car'])

df.columns
df['Car']
Reporte_Nulos(df)
imputacion_basico(df, 'mean', ['Car'],group=['Suburb']) 
import time
t0 = time.time()
Borrar_Nulos(df)
t1 = time.time()

total = t1-t0
print(total)
t0 = time.time()
nullity_filter(df,filter='top', p=0.9)
t1 = time.time()

total = t1-t0
print(total)