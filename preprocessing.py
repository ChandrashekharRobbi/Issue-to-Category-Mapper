import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

def vectorizer():
    category_mapping = {'Debt collection': 0, 
                        'Mortgage': 1, 
                        'Consumer Loan': 2, 
                        'Bank account or service': 3, 
                        'Credit reporting': 4, 
                        'Payday loan': 5, 
                        'Other financial service': 6, 
                        'Student loan': 7, 
                        'Money transfers': 8, 
                        'Prepaid card': 9, 
                        'Credit card': 10
                        }
    
    print("Created Mapping")
    model = pickle.load(open("model.pkl",'rb'))
    vectorizer = pickle.load(open("vectorizer.pkl",'rb'))
    print("Loaded the model")
    return model, vectorizer, category_mapping