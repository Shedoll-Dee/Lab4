import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

def download_data():
    df = pd.read_csv('./adult.csv', delimiter = ',')
    df.to_csv("cars.csv", index = False)
    return df

def clear_data(path2df):
    df = pd.read_csv(path2df)
    
    print(df.columns)

    cat_columns = ['age', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week', 'native.country']
    num_columns = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'income']

    df = df.drop('fnlwgt')

    print("Обработка пропущенных значений...")
    
    # Заменяем "?" на NaN
    df = df.replace('?', np.nan)

    initial_rows = len(df)
    df = df.dropna()
    removed_rows = initial_rows - len(df)
    
    print(f"Удалено строк с пропусками: {removed_rows}")
    print(f"Осталось строк: {len(df_processed)}")
    
    df.to_csv('df_clear.csv')
    return True

download_data()
clear_data("cars.csv")
