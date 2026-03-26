import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

def download_data():
    df = pd.read_csv('./adult.csv', delimiter = ',')
    df.to_csv("cars.csv", index = False)
    return df

def clear_data(path2df):
    df = pd.read_csv(path2df)
    
    print(df.columns)
    df.to_csv('df_clear.csv')
    return True

download_data()
clear_data("cars.csv")
