import pandas as pd
import numpy as np
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.preprocessing import LabelEncoder


def download_data():
    print('Начало скачивания датасета')
    api = KaggleApi()
    api.authenticate()

    api.dataset_download_files(
        "uciml/adult-census-income",
        path=".",
        unzip=True
    )
    df = pd.read_csv("./adult.csv", delimiter = ',')
    return df


def clear_data():
    df = download_data()
    
    print(df.columns)

    cat_columns = ['workclass', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
    num_columns = ['age', 'capital.gain', 'capital.loss', 'hours.per.week', 'education.num']
    
    df = df.drop(['fnlwgt'], axis=1)
    df = df.drop(['education'], axis=1)

    df['income'] = np.where(df['income'] == '<=50K', 0, 1)

    print("Обработка пропущенных значений...")
    
    # Заменяем "?" на NaN
    df = df.replace('?', np.nan)

    initial_rows = len(df)
    df = df.dropna()
    removed_rows = initial_rows - len(df)
    
    print(f"Удалено строк с пропусками: {removed_rows}")
    print(f"Осталось строк: {len(df)}")
    
    df['native.country'] = np.where(df['native.country'] == 'United-States', 'United-States', 'Other')
    
    # кодирование данных
    for col in cat_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    
    df.to_csv('df_clear.csv', index=False)
    return True

clear_data()
