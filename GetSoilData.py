import pandas as pd
import numpy as np


df = pd.read_csv('SoilHealthDB_V1.csv', encoding='ISO-8859-1')
l = ['Country', 'N_C', 'K_C', 'P_C', 'pH_C']
df2 = df[l]


def getData(country):
    l2 = l[1:]
    df3 = df2[df2['Country'] == country]
    df3.dropna()
    col_mean = df3[l2].mean()
    a = col_mean
    a = {

    }
    a['N'] = col_mean['N_C']
    a['K'] = col_mean['K_C']
    a['P'] = col_mean['P_C']
    a['pH'] = col_mean['pH_C']
    return a


print(getData('USA'))
