# author: sarbhanub
# link: github.com/sarbhanub
# date: 04.09.2023
# competition: playground-series-s3e12
# task: binary classification, balanced classes

# essentials
import pandas as pd
import numpy as np
# scalers
from sklearn.preprocessing import StandardScaler
# models
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
# from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
# ensembler
from sklearn.ensemble import VotingClassifier
# hyperparameter optimizers
from functools import partial
from hyperopt import hp, Trials, fmin, tpe
# metrics and cross validation
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# random and static options
rs = 17 # random state
scaler = StandardScaler() # data scaler
splits = 10 # KFold splits for all our cross-validation tests
max_evals = 25 # number of trials for hyperopt

# defining models
models = {'Random Forest': RandomForestClassifier(random_state=rs),
          'XGB Classifier': XGBClassifier(booster='gbtree', tree_method='gpu_hist', predictor='gpu_predictor',sampling_method='gradient_based',grow_policy='lossguide',random_state=rs),
          'LGBM Classifier': LGBMClassifier(objective='binary',boosting_type='gbdt',random_state=rs),
        #   'CatBoost Classifier': CatBoostClassifier(verbose=False,random_state=rs),
          'Logistic Regression': LogisticRegression(solver='saga',penalty='elasticnet',class_weight=None,dual=False,fit_intercept=True,intercept_scaling=1,random_state=rs),
          'SVMC': SVC(probability=True, random_state=rs)}

# param spaces for each defined model
param_space = {'Random Forest': {'n_estimators': hp.randint('n_estimators', 700, 1200),
                                 'max_depth': hp.randint('max_depth', 3, 13),
                                 'min_samples_split': hp.randint('min_samples_split', 3, 7),
                                 'min_samples_leaf': hp.randint('min_samples_leaf', 3, 13),
                                 'max_samples': hp.uniform('max_samples', 0.49, 0.99)},

                'XGB Classifier': {'n_estimators': hp.randint('n_estimators', 600, 1200),
                                    'learning_rate': hp.uniform('learning_rate', 0.01, 0.44),
                                    'max_depth': hp.randint('max_depth', 4, 12),
                                    'min_child_weight': hp.randint('min_child_weight', 2, 12),
                                    'subsample': hp.uniform('subsample', 0.85, 1),
                                    'reg_alpha': hp.uniform('reg_alpha', 0.01, 0.5),
                                    'reg_lambda': hp.uniform('reg_lambda', 1.0, 3.0),
                                    'max_delta_step': hp.randint('max_delta_step', 2, 10)},

                'LGBM Classifier': {'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
                                    'subsample': hp.uniform('subsample', 0.1, 1),
                                    'colsample_bytree': hp.uniform('colsample_bytree', 0.1, 1),
                                    'reg_alpha': hp.uniform('reg_alpha', 0.01, 0.09),
                                    'reg_lambda': hp.uniform('reg_lambda', 0.001, 0.09),
                                    'min_child_samples': hp.randint('min_child_samples', 3, 23),
                                    'max_depth': hp.randint('max_depth', 3, 15),
                                    'n_estimators': hp.randint('n_estimators', 200, 900)},

                # 'CatBoost Classifier': {'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
                #                         'depth': hp.quniform('depth', 4, 10, 1),
                #                         'l2_leaf_reg': hp.uniform('l2_leaf_reg', 0, 10),
                #                         'iterations': hp.quniform('iterations', 100, 1000, 100),
                #                         'border_count': hp.quniform('border_count', 32, 255, 1),
                #                         'bagging_temperature': hp.uniform('bagging_temperature', 0.1, 0.99),
                #                         'random_strength': hp.uniform('random_strength', 0.1, 0.99),
                #                         'scale_pos_weight': hp.uniform('scale_pos_weight', 0, 4)},

                'Logistic Regression': {'C': hp.uniform('C', 0.1, 2.5),
                                        'max_iter': hp.randint('max_iter', 800, 1200),
                                        'l1_ratio': hp.uniform('l1_ratio', 0.01, 0.99),
                                        'tol': hp.loguniform('tol', -5, 2)},

                'SVMC': {'C': hp.uniform('C', 0.0, 0.7),
                        'gamma': hp.uniform('gamma', 0, 1),
                        'tol': hp.loguniform('tol', -5, 2)}}

# data import
train = pd.read_csv("input/playground-series-s3e12/train.csv")
test = pd.read_csv("input/playground-series-s3e12/test.csv")

# setting features and label
X, y = train.iloc[:,1:-1], train.iloc[:,-1]
X_test = test.iloc[:, 1:]

# data scaling for both train and test
X = scaler.fit_transform(X)
X_test = scaler.fit_transform(X_test)

# cross-validation config
skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=rs)

# printing cv details
fold = 1
for train_idx, valid_idx in skf.split(X, y):
    print(f'fold {fold}: train: {len(train_idx)}; validation:{len(valid_idx)}')
    fold+=1

print('\n')
# hyperopt tuning function //where the magic happens
def hyperopt_tuning(X, y, models, param_space, cv, max_evals=30):
    best_params = {}
    for name, model in models.items():
        print(f"Hyperparameter tuning for {name}...")
        def optimize(params, X, y):
            model.set_params(**params) # *** v important ***
            scores = []
            for idx in cv.split(X=X, y=y):
                # train_valid split for each fold/split
                train_idx, valid_idx = idx[0], idx[1]
                xtrain, xvalid = X[train_idx], X[valid_idx]
                ytrain, yvalid = y[train_idx], y[valid_idx]
                # fitting the fold
                model.fit(xtrain, ytrain)
                # preds for fold
                preds = model.predict(xvalid)
                fold_acc = roc_auc_score(yvalid, preds)
                scores.append(fold_acc)

            return -1*np.mean(scores)

        # deining trial for telemetry
        trials = Trials()
        # optimization function
        optimization_function = partial(optimize, X=X, y=y)
        # saving best parameters
        best = fmin(fn=optimization_function, space=param_space[name], algo=tpe.suggest, max_evals=max_evals, trials=trials)
        best_params[name] = best
        
    return best_params

# running the hyperparameter tuning function
best_params = hyperopt_tuning(X=X, y=y, models=models, param_space=param_space, cv=skf, max_evals=max_evals)


# using voting classifier

# creating an ensemble of the models using soft voting
estimators = [('Random Forest', RandomForestClassifier(**best_params['Random Forest'],random_state=rs)),
              ('XGB Classifier', XGBClassifier(**best_params['XGB Classifier'],booster='gbtree',tree_method='gpu_hist',predictor='gpu_predictor',sampling_method='gradient_based',grow_policy='lossguide',random_state=rs)),
              ('LGBM Classifier', LGBMClassifier(**best_params['LGBM Classifier'], objective='binary',boosting_type='gbdt',random_state=rs)),
               #   ('CatBoost Classifier', CatBoostClassifier(**best_params['CatBoost Classifier'],verbose=False,random_state=rs)),
              ('Logistic Regression', LogisticRegression(**best_params['Logistic Regression'],solver='saga',penalty='elasticnet',class_weight=None,dual=False,fit_intercept=True,intercept_scaling=1,random_state=rs)),
              ('SVMC', SVC(**best_params['SVMC'],probability=True,random_state=rs))]

ensembler = VotingClassifier(estimators=estimators, voting='soft')

# printing the data for verification
for estimator in ensembler.estimators:
    print(f"Model: {estimator[0]}")
    print(f"Parameters: {estimator[1].get_params()}")
    print()

# checking the cv score for our ensamble as well to understand if it's better or worse
ens_scores = []
for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X[train_idx], y[train_idx]
    X_valid, y_valid = X[valid_idx], y[valid_idx]
    ensembler.fit(X, y)
    preds = ensembler.predict(X_valid)
    fold_roc_auc = roc_auc_score(y_valid, preds)
    ens_scores.append(fold_roc_auc)

    # print the importance and weight of each cv and model
    for name, estimator in ensembler.estimators:
        importance = ensembler.named_estimators_[name].score(X_valid, y_valid)
        weight = ensembler.named_estimators_[name].n_features_in_
        print(f"fold {fold} | {name}: importance={importance:.4f}, weight={weight}")

    print()

print('Final ensemble score with CV:'+str(np.mean(ens_scores)))

# using stacked meta classifier
##############################


# training final model if happy with score
ensembler.fit(X, y)

# final prediction
preds = ensembler.predict_proba(X_test)

# kaggle submission file
submission = pd.DataFrame({'id': test.id, 'target': preds[:,1]})
submission.to_csv('submission.csv', index=False)