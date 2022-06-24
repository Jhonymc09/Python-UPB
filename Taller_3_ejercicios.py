# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 16:01:39 2022

@author: Jhony Meléndez
"""
#Diplomado Python aplicado a la ingeniería
#Autor Jhony Meléndez
#ID 502198
#Jhony.melendez@upb.edu.co

import pandas as pd
from datetime import datetime
import numpy as np

info = pd.DataFrame({'NOMBRE': ['ELSY SILGADO','DEIMER MORELO','JESUS PORTILLO',
                                 'JONNIER TERAN','JONATHAN BRITO','GUSTAVO BLANCO',
                                 'DAYAN HERNANDEZ','WENDY MENDOZA','ALVARO ZAMORA',
                                 'GABRIELA RAMOS','MIGUEL FABRIS','JHONY MELENDEZ',
                                 'ADRIANA SUAREZ','JAIDER ECHEVERRI','NOIVER DARIO RAMOS',
                                 'MIRIAM LOPEZ','IVAN PALENCIA','ANGELA POSSO'],
                     'EDAD': [23,25,21,26,22,26,28,29,21,22,27,26,27,22,26,23,22,27],
                     'SEXO': ['F','M','M','M','M','M','F','F','M','F','M','M',
                              'F','M','M','F','M','F'],
                     'PESO': [64,72,71,75,77,71,63,60,70,62,73,79,60,73,71,67,72,67],
                     'ALTURA': [160,172,171,175,172,160,165,171,178,158,190,150,
                                165,172,176,177,171,174],
                     'DINERO_INVERTIR': [30000000,2000000,4000000,2000000,6000000,30000000,
                                         6000000,3000000,6000000,9000000,14000000,20000000,
                                         42875000,10000000,30000000,3000000,10000000,20000000],
                     'INTERES_ANUAL': [0.04,0.04,0.06,0.06,0.04,0.05,0.06,0.07,0.07,
                                       0.06,0.05,0.05,0.06,0.05,0.06,0.06,0.06,0.07],
                     'ANOS_INVERSION': [1,2,3,3,2,1,1,2,3,3,2,1,1,2,3,3,2,1],
                     'TELEFONO': ['3122569447','3116364059','3123063546','3236665257',
                                  '3027482936','3176765047','3155691850','3001461692',
                                  '3001838809','3232792344','3038615222','3001688142',
                                  '3017510586','3108741997','3343112735','3005459861',
                                  '3008737611','3006792711'],
                     'HORA_COMPRA_PAN':['5:00:00','6:00:00','8:10:00','12:00:00','18:30:00',
                                        '18:00:00','9:00:00','7:15:00','12:20:00','16:30:00',
                                        '20:30:00','8:30:00','15:00:00','8:10:00','14:30:00',
                                        '9:00:00','10:00:00','8:00:00']})

#EJERCICIO 1
for i in info.index:
    m = round(((info["PESO"][i])/((info["ALTURA"][i])/100)),2)
    print("""Saludos... {} Tu masa corporal es de: {} KG""".format(str(info["NOMBRE"][i]),str(m)))
    print("---------------------------------------------------")
    
# EJERCICIO 2
for i in info.index:
    capital = round(info["DINERO_INVERTIR"][i]*(((info["INTERES_ANUAL"][i])
                                                 /(100+1))**info["ANOS_INVERSION"][i]),2)
    print("""Saludos... {} el capital a invertir es de: $ {}""".format(str(info["NOMBRE"][i])
                                                                      ,str(capital)))
    print("---------------------------------------------------")

#EJERCICIO 3
info["HORAS_HORN"] = ((pd.to_timedelta(info["HORA_COMPRA_PAN"])-pd.to_timedelta("5:15:00")).dt.total_seconds())//3600

condic = [
    (info["HORAS_HORN"] >= 2) & (info["HORAS_HORN"]<5),
    (info["HORAS_HORN"] >= 3) & (info["HORAS_HORN"]<8),
    (info["HORAS_HORN"] <= 10) & (info["HORAS_HORN"]<15),
    (info["HORAS_HORN"] <= 12) & (info["HORAS_HORN"]<20)]
selec = [0.3, 0.2, 0.1, 0.5]

info["PORCENTAJE_DESC"] =  np.select(condic, selec, default='Not Specified')
info["PRECIO"] = 7000-(pd.to_numeric(info["PORCENTAJE_DESC"], errors='coerce') *7000)
print(info["PRECIO"])

# EJERCICIO 4
condic_ext = [
    (info["SEXO"] == "F"),
    (info["SEXO"] == "M")]
selec_ext = [8, 10]


info["EXT_CEL"] = np.select(condic_ext, selec_ext, default='Not Specified')
print(info["EXT_CEL"])
