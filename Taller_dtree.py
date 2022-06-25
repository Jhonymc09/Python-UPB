# -*- coding: utf-8 -*-
"""
@author: Jhony Meléndez
# 
"""

#Diplomado Python aplicado a la ingeniería
#Autor Jhony Meléndez
#ID 502198
#Jhony.melendez@upb.edu.co

#Importamoms las librerias

import statsmodels.api as sm
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
import pydotplus


cars = sm.datasets.get_rdataset("Carseats", "ISLR")
info = cars.data
print(cars.__doc__)
info['ventas_altas'] = np.where(info.Sales > 8, 0, 1)
info = info.drop(columns = 'Sales')

d= {'Yes':1, 'No':0}
d1= {'Bad':0, 'Medium':1, 'Good':2}
info["Urban"] = info["Urban"].map(d)
info["US"] = info["US"].map(d)
info["ShelveLoc"] = info["ShelveLoc"].map(d1)

f = ["CompPrice","Income","Advertising", "Population","Price","ShelveLoc","Age","Education","Urban","US"]

X = info[f]
y = info["ventas_altas"]

xtrain = X[:320]
ytrain = y[:320]

xtest = X[320:]
ytest = y[320:]

dtree = DecisionTreeClassifier()
dtree  = dtree.fit(xtrain, ytrain)

prediction = dtree.predict([[136,70,12,171,152,1,44,18,1,1]])

info = tree.export_graphviz(dtree, out_file = None, f_names= f)
graph = pydotplus.graph_from_dot_data(info)
graph.write_png('mydecisiontree.png')
img = pltimg.imread("mydecisiontree.png")
imgplot = plt.imshow(img)
plt.show()