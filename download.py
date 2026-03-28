import pandas as pd
import numpy as np


def clear_data():
    df = pd.read_csv('./adult.csv', delimiter = ',')
    
    print(df.columns)

    cat_columns = ['workclass', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
    num_columns = ['age', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week', 'education.num']
    
    df = df.drop(['fnlwgt'], axis=1)

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
    
    # one-hot кодирование данных
    df = pd.get_dummies(df, columns=cat_columns)
    
    df.to_csv('df_clear.csv')
    return True

clear_data()
