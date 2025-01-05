import pandas as pd

import kerastuner as kt

from sklearn import metrics, model_selection

# from auto_models import AutoModel
# from tuner import SklearnCVTuner

from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier

from imblearn import combine, over_sampling, under_sampling, pipeline

# import pickle

def micro_f1(y_true, y_pred):
    return metrics.f1_score(y_true, y_pred, average="micro")

def macro_f1(y_true, y_pred):
    return metrics.f1_score(y_true, y_pred, average="macro")


if __name__ == "__main__":
    
    rand_seeds = range(51)
    column_names = ["dataset", "random_seed", "f1_micro", "f1_macro", "accuracy", 'classification_report']
    results_df = pd.DataFrame(columns = column_names)

    # code_smell = 'feature_envy'
    code_smell = 'data_class'

    data_path_base = code_smell + '/data/embedded_datasets/'

    for rand_seed in rand_seeds:
        # print('*'*10 + str(rand_seed) + '*'*10)

        y_test_ids = pd.read_csv(code_smell + "/data/data_splits/y_test_" + str(rand_seed) + ".csv")
        y_train_ids = pd.read_csv(code_smell + "/data/data_splits/y_train_" + str(rand_seed) + ".csv")

        data_paths = [
            "metrics_pmd_merged.pkl",
            # "whole_pmd_merged.pkl",
            # "sum_pmd_merged.pkl",
            # "avg_pmd_merged.pkl",
            ]

        for data_path in (data_paths):
            # print('-'*10 + data_path + '-'*10)
            data = pd.read_pickle(data_path_base + data_path)
            train = data.loc[data['sample_id'].isin(y_train_ids['sample_id'])]
            test = data.loc[data['sample_id'].isin(y_test_ids['sample_id'])]
            # data = pd.read_csv("../data/metrics_dataset.csv")
            try:
                X_train_df = train.drop(columns=['label', 'sample_id', 'severity'])
                # print(X_train_df.head())
                X_test_df = test.drop(columns=['label', 'sample_id', 'severity'])
            except Exception as e:
                X_train_df = train.drop(columns=['label', 'sample_id'])
                X_test_df = test.drop(columns=['label', 'sample_id'])

            # data class metrics dataset
            try:
                X_train_df = X_train_df.drop(columns=['lcc', 'tcc'])
                X_test_df = X_test_df.drop(columns=['lcc','tcc'])
            except:
                pass

            y_train_df = train.label
            y_test_df = test.label

            X_train = X_train_df.values
            y_train = y_train_df.values
            X_test = X_test_df.values
            y_test = y_test_df.values
            
            best_model = pipeline.make_pipeline(
                StandardScaler(),
                combine.SMOTEENN(
                    random_state=42,
                    smote=over_sampling.SMOTE(random_state=42),
                    enn=under_sampling.EditedNearestNeighbours(
                        sampling_strategy="majority",
                        kind_sel="mode"
                    ),
                ),
                XGBClassifier(
                    n_estimators=500,
                    # use_label_encoder=False,
                    eval_metric="logloss",
                    learning_rate=0.1,
                    max_depth=6,
                    subsample=0.6,
                    # gamma=0,
                    min_child_weight=3,
                    # colsample_bytree=0.8,
                    # colsample_bylevel=0.8,
                    # colsample_bynode=0.6,
                    reg_lambda=1,
                    # reg_alpha=hp.Float("reg_alpha", 0, 10, sampling="linear"),
                    seed=42,
                    # n_jobs=-1
                )
            )
            
            best_model = best_model.fit(X_train, y_train)

            y_pred_train = best_model.predict(X_train)
            y_pred = best_model.predict(X_test)

            print(f'----------------------------------    RESULTS: {rand_seed}/{len(rand_seeds)} ' + data_path + '---------------------------------------')

            f1_micro =  micro_f1(y_train, y_pred_train)
            f1_macro = macro_f1(y_train, y_pred_train)
            accuracy = metrics.accuracy_score(y_train, y_pred_train)
            report = metrics.classification_report(y_test, y_pred)
            print("\nTrain metrics")
            print("Train micro f1: ", f1_micro)
            print("Train macro f1:", f1_macro)
            print("Train accuracy: ", accuracy)
            print("\nTest metrics")
            print("Test report: \n", report)
            print(f"Seed: {rand_seed}/{len(rand_seeds)}")

            new_row = {'dataset': data_path, 'random_seed': rand_seed, 'f1_micro': f1_micro, 'f1_macro':f1_macro, 'accuracy':accuracy, 'classification_report':report}
            results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)

    results_df.to_pickle(code_smell + '/results/results.pkl')
    results_df.to_csv(code_smell + '/results/results.csv')
