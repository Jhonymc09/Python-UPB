# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 13:00:27 2022

@author: Jhony Meléndez
"""
#Diplomado Python aplicado a la ingeniería
#Autor Jhony Meléndez
#ID 502198
#Jhony.melendez@upb.edu.co

# import librerias
import pandas as pd
import numpy as np
from sklearn import linear_model

# 1) A partir de los valores independientes (volumen, peso y producción de CO2)
#predecir el comportamiento de la variable dependiente (marca del carro.)

# leemos archivo csv con pandas y creamos df
df = pd.read_csv('cars.csv')

r = [(df["Car"] == "Audi"),
    (df["Car"] == "BMW"),
    (df["Car"] == "Fiat"),
    (df["Car"] == "Ford"),
    (df["Car"] == "Honda"),
    (df["Car"] == "Hundai"),
    (df["Car"] == "Mazda"),
    (df["Car"] == "Mercedes"),
    (df["Car"] == "Mini"),
    (df["Car"] == "Mitsubishi"),
    (df["Car"] == "Opel"),
    (df["Car"] == "Skoda"),
    (df["Car"] == "Suzuki"),
    (df["Car"] == "Toyoty"),
    (df["Car"] == "VW"),
    (df["Car"] == "Volvo"),
    (df["Car"] == "Hyundai")]
opc = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0]

df["ICar"] =  np.select(r, opc, default='Not Specified')
pd.to_numeric(df["ICar"])

# hacer una lista de variables independientes
x = df[["Volume","Weight","CO2"]]

# Variable Dependiente
y = df["ICar"]

# Regresion
reg_mod = linear_model.LinearRegression()
reg_mod.fit(x,y)

#Prediccion
predict_co2 = reg_mod.predict([[1000,790,99]])
print(predict_co2)
print(reg_mod.coef_)