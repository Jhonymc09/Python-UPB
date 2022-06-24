# -*- coding: utf-8 -*-
"""
Created on Sat May 28 11:14:37 2022

@author: Jhony Meléndez
# 
"""

#Diplomado Python aplicado a la ingeniería
#Autor Jhony Meléndez
#ID 502198
#Jhony.melendez@upb.edu.co

a=1
b=2
c=3
d=4
e=5
f=6

ecu1 = (a+(b/c))/(d+(e/f))
Ecu2 = a-(b/(c-d))

Ecu3 = ecu1
ecu1 = Ecu2
Ecu2 = Ecu3

print(ecu1)
print(Ecu2)