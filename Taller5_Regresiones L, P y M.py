# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 16:07:55 2022

@author: user
"""

#Realizar una regresión lineal
#polinomial y múltiple con los datos asociados.
#Las variables escogidas son libres.

#Importamos librerias
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

#Leemos el archivo csv con pandas y creamos df
df = pd.read_excel('AirQualityUCI.xlsx')

x = df["CO(GT)"]
y = df["NO2(GT)"]

slope,intercept,r,p,std_err = stats.linregress(x,y)

def regresion(x):
    return slope*x + intercept

r = list(map(regresion, x))

plt.scatter(x,y)
plt.plot(x,r)
plt.show()

#Prediccion Regresion Lineal
predict_reg_lineal = regresion(3)
print("""La prediccion es {} y el r de relacion es {}""".format(str(predict_reg_lineal), str(r)))


poli_model = np.poly1d(np.polyfit(x,y, 5))

poli_line = np.linspace(1,18,100)

poli_new_y = poli_model(poli_line)

plt.scatter(x,y)
plt.plot(poli_line, poli_new_y)
plt.show()

#Prediccion Polimonial
predict_reg_poli = poli_model(3)
r2_poli = r2_score(y,poli_model(x))
print("""La prediccion es {} y el r de relacion es {}""".format(str(predict_reg_poli), str(r2_poli)))


z = df[["CO(GT)"]]

reg_mod = linear_model.LinearRegression()
reg_mod.fit(z,y)

#Prediccion Multiple
predict_multi = reg_mod.predict([[3]])
r_multi = reg_mod.coef_

print("""La prediccion es {} y el r de relacion es {}""".format(str(predict_multi), str(r_multi)))
