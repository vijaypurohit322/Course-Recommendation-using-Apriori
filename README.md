# Course-Recommendation-using-Apriori
In this project, I've built a model to recommend courses to the users on the basis of their interest using Apriori Algorithm.

library used while building this project are:

import numpy as np 

import pandas as pd 

from mlxtend.frequent_patterns import apriori, association_rules 

from mlxtend.preprocessing import TransactionEncoder

Later on I have Use Hot encoding to make it right for model


def hot_encode(x): 
    if(x<= 0): 
        return 0
    if(x>= 1): 
        return 1
  
# Encoding the datasets 

SP_encoded = SPECIALIZATION.applymap(hot_encode) 

SPECIALIZATION = SP_encoded 
  

C_encoded = COURSE.applymap(hot_encode) 

COURSE = C_encoded 
  

P_encoded = PROFESSIONAL.applymap(hot_encode) 

PROFESSIONAL = P_encoded

