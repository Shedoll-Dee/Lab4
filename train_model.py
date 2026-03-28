from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
import mlflow
from mlflow.models import infer_signature
import joblib


def eval_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    f1 = f1_score(actual, pred)
    precision = precision_score(actual, pred)
    recall = recall_score(actual, pred)
    return accuracy, f1, precision, recall


if __name__ == "__main__":
    # Загружаем очищенный DataFrame
    df = pd.read_csv("./df_clear.csv")

    # X — все признаки, Y — целевая переменная
    X = df.drop(columns=['income'])
    Y = df['income']

    # Разделяем на train и validation
    X_train, X_val, y_train, y_val = train_test_split(
        X.values,  # оставляем numpy, без масштабирования/one-hot
        Y.values,
        test_size=0.3,
        random_state=42
    )

    # Параметры GridSearchCV
    params = {
        'alpha': [0.0001, 0.001, 0.01],
        'l1_ratio': [0.0, 0.15, 0.5],
        "penalty": ["l1", "l2", "elasticnet"],
        "loss": ["hinge", "log_loss"],
        "fit_intercept": [True, False]
    }

    mlflow.set_experiment("adult income classification")

    with mlflow.start_run():
        model = SGDClassifier(random_state=42)

        # Подбираем лучшие параметры
        clf = GridSearchCV(
            model,
            params,
            cv=3,
            n_jobs=4
        )
        clf.fit(X_train, y_train)
        best = clf.best_estimator_

        # Предсказание на валидации
        y_pred = best.predict(X_val)
        accuracy, f1, precision, recall = eval_metrics(y_val, y_pred)

        # Логируем параметры и метрики
        mlflow.log_param("alpha", best.alpha)
        mlflow.log_param("l1_ratio", best.l1_ratio)
        mlflow.log_param("penalty", best.penalty)
        mlflow.log_param("loss", best.loss)
        mlflow.log_param("fit_intercept", best.fit_intercept)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        # Логируем модель с signature ровно по исходным 12 колонкам
        signature = infer_signature(X, best.predict(X.values))
        mlflow.sklearn.log_model(best, "model", signature=signature)

        # Сохраняем pickle локально
        with open("adult_model.pkl", "wb") as file:
            joblib.dump(best, file)

    # Получаем путь к лучшей модели
    dfruns = mlflow.search_runs()
    path2model = (
        dfruns
        .sort_values("metrics.f1", ascending=False)
        .iloc[0]['artifact_uri']
        .replace("file://", "") + '/model'
    )
    print(path2model)
