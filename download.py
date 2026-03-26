import pandas as pd
import numpy as np


def clear_data():
    df = pd.read_csv('./adult.csv', delimiter = ',')
    
    print(df.columns)

    cat_columns = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'income']
    num_columns = ['age', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week', 'native.country']

    df = df.drop(['fnlwgt'], axis=1)

    print("Обработка пропущенных значений...")
    
    # Заменяем "?" на NaN
    df = df.replace('?', np.nan)

    initial_rows = len(df)
    df = df.dropna()
    removed_rows = initial_rows - len(df)
    
    print(f"Удалено строк с пропусками: {removed_rows}")
    print(f"Осталось строк: {len(df)}")

    # one-hot кодирование данных
    df = pd.get_dummies(df, columns=cat_columns)
    
    df.to_csv('df_clear.csv')
    return True

clear_data()
