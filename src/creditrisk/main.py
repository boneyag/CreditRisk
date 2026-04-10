from src.creditrisk import data
from src.creditrisk import preprocess
from src.creditrisk import train
from src.creditrisk import evaluate

def main(model_name="lr"):
    df = data.load_typecast_data()

    if df is not None:
        X, y = preprocess.data_target_split(df)
        X = preprocess.feature_engineering(X)
        X_train, X_test, y_train, y_test = preprocess.partition_data(X, y)
        preprocessor = preprocess.preprocessing_pipeline(X_train)
        clf = train.train_classifier(model_name, preprocessor, X_train, y_train, persist=True)
        eval_metrics = evaluate.eval_classifier(model_name, X_test, y_test, clf)

        print(f"{model_name} evaluation metrics:\n{eval_metrics}")

if __name__ == '__main__':
    main("xgb")