
# MIT License
#
# Copyright (c) 2024 Hongzhou Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# ...



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
# from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif']=['SimHei']
from sklearn.ensemble import RandomForestClassifier
import optuna
def R_score(x_train, y_train):
    R=pd.Series(x_train).corr(pd.Series(y_train))**2
    return R
file_path = r"..\DATA\Final_RF_Inward.xlsx"
Inward= pd.read_excel(file_path, sheet_name="Sheet1")
train_data_y=Inward["SOS_Diff_ABS"]
train_data_X01=Inward.iloc[:,1:10]
def R_score(x_train, y_train):
    R = pd.Series(x_train).corr(pd.Series(y_train)) ** 2
    return R
def KGE(q1, q2):
    error = []
    for i in range(len(q1)):
        error.append(q2[i] - q1[i])
    squaredError = []
    for val in error:
        squaredError.append(val * val)
    aDeviation = []
    aMean = sum(q2) / len(q2)
    for val in q2:
        aDeviation.append((val - aMean) * (val - aMean))
    NSE = 1 - sum(squaredError) / sum(aDeviation)
    return NSE

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
print(np.array(train_data_y))
print(Normalizer().fit_transform(np.array(train_data_y).reshape(-1, 1)).flatten())
Scaler=MinMaxScaler()
Scaler2=Normalizer()
train_data_X=Scaler.fit_transform(np.array(train_data_X01))

importances001=[]
def objective(trial):
                param = {
                    'n_estimators': trial.suggest_int('n_estimators', 5, 2000),
                    "max_depth":trial.suggest_categorical('max_depth',[5,10,100,200,150,80,90,70,300,320,350,360,400]),
                    "max_features": trial.suggest_int("max_features", 1, 8),
                    "min_samples_leaf":trial.suggest_int("min_sample_leaf", 2, 11),
                    "random_state":202
                }
                forest = sklearn.ensemble.RandomForestRegressor(**param)
                forest.fit(train_data_X, train_data_y)
                pred_lgb1 = forest.predict(train_data_X)
                r2_score1 = R_score(train_data_y, pred_lgb1)
                importances = forest.feature_importances_
                importances001.append(importances)
                return r2_score1
study=optuna.create_study(direction='maximize', sampler=optuna.samplers.RandomSampler())
n_trials=100
study.optimize(objective, n_trials=n_trials)
best_trial = study.best_trial
print(best_trial)
DF=pd.DataFrame(np.array(importances001).reshape(100,9),
    columns=Inward.iloc[:,1:10].columns).to_csv(r"..\DATA\\In_SOS_impor.csv")