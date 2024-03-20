import numpy as np
import pandas as pd

import time

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

from datetime import datetime, timedelta
import re
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import webtestapp.preprocessing_mojule as pm
import pickle
from sklearn.metrics import mean_squared_error
import os
from webtestapp.forms import MyForm

pickle_dir = "static/pickle"

#result_addinfo_path = os.path.join(pickle_dir, "result_addinfo.pickle")
#horse_history_path = os.path.join(pickle_dir, "horse_history.pickle")
#jockey_history_path = os.path.join(pickle_dir, "jockey_history.pickle")
pay_dict_path = os.path.join(pickle_dir, "pay_dict.pickle")
#dict_horse_ped_dict_path = os.path.join(pickle_dir, "dict_horse_ped.pickle")
data_path = os.path.join(pickle_dir, "data.pickle")

#results_addinfo = pd.read_pickle(result_addinfo_path)
#horse_history = pd.read_pickle(horse_history_path)
#jockey_history = pd.read_pickle(jockey_history_path)
#dict_horse_history = {idx: horse_history.loc[idx] for idx in horse_history.index.unique()}

with open(pay_dict_path, "rb") as f:
    pay_dict = pickle.load(f)

"""    
with open(dict_horse_ped_dict_path, "rb") as f:
    horse_ped_dict = pickle.load(f)
"""
    
class Results:
    def __init__(self, pre_df):
        self.pre_df = pre_df
        self.before_scale = None
        self.post_df = self.preprocessing(self.pre_df)
        
    
    def preprocessing(self, pre_df):
        results = pre_df.copy()
        results = results[~(results["着順"].astype(str).str.contains("\D"))]
        results["着順"] = results["着順"].astype(int)
        results["性齢"] = results["性齢"].astype(str)
        results["性"] = results["性齢"].map(lambda x:x[0])
        results["年齢"] = results["性齢"].map(lambda x:x[1:])
        results["体重"] = results["馬体重"].map(lambda x:x[:3])
        results["増減"] = results["馬体重"].str.split("(").map(lambda x:int(x[-1][:-1]))
        results["人気"] = results["人気"].astype(int)
        results["年齢"] = results["年齢"].astype(int)
        results["体重"] = results["体重"].astype(float)
        results["単勝"] = results["単勝"].astype(float)
        results = pd.concat([results, pd.get_dummies(results["性"])], axis=1)
        results["勝率*騎乗回数"] = results["勝率"] * results["騎乗回数"]
        results["連対率*騎乗回数"] = results["連対率"] * results["騎乗回数"]
        results["複勝率*騎乗回数"] = results["複勝率"] * results["騎乗回数"]
        
        
        results["増減/体重"] = results["増減"] / results["体重"]  
        results["斤量/体重"] = results["斤量"] / results["体重"]
        
        features_addinfo = [
            '着順', '枠番', '馬番', '馬名', '性齢', '斤量', '騎手', 'タイム', '着差', '単勝', '人気',
       '馬体重', '調教師', 'horse_id', 'jockey_id', 'length', 'race_type', 'weather',
       'condition', 'date', '騎乗回数', '勝率', '連対率', '複勝率', '賞金_ave', '賞金_sum',
       '順番_ave', '賞金_ave_2', '賞金_sum_2', '順番_ave_2', '賞金_ave_4', '賞金_sum_4',
       '順番_ave_4', '順番_ave_distorted', '順番_ave_2_distorted',
       '順番_ave_4_distorted', '芝', 'ダ', '障', '短距離', 'マイル距離', '芝_ave_order',
       'ダ_ave_order', '短距離_ave_order', 'マイル距離_ave_order', '芝得意', '短距離得意',
       '中距離_ave_order', '長距離_ave_order', 'マイル距離得意', '中距離得意', '長距離得意',
       'length_match', 'race_type_match', 'horse_ped_score', 'タイム_線形',
       'score', 
        ]
        
        drop_features = [
            "枠番", "馬名", "性齢", "騎手","着差", "馬体重", "調教師", "horse_id",
            "jockey_id",  "race_type", "weather", "condition", "date", "性",
            '芝', 'ダ', '障', '短距離', 'マイル距離', '芝_ave_order',
            'ダ_ave_order', '短距離_ave_order', 'マイル距離_ave_order', '芝得意', '短距離得意',
            '中距離_ave_order', '長距離_ave_order', 'マイル距離得意', '中距離得意', '長距離得意',
            "タイム", "タイム_線形", '賞金_ave_4', '賞金_sum_4','順番_ave_4', '順番_ave_distorted', '順番_ave_2_distorted',
            '順番_ave_4_distorted', 
        ]
        results = results.drop(drop_features, axis=1)
        results = results.astype(float)
        results["着順"] = results["着順"].astype(int)
        self.before_scale = results
        
        keep_features = ['着順', 'セ', '牝', '牡', "馬番", "length_match", "race_type_match", "人気", "単勝", "score"]
        scaler = pm.CustomSTDScaler(keep_features)
        results = scaler.fit_transform(results)
        return results


class Cal:
    @staticmethod
    def cal_tansho(x):
        race_id = x.name
        horse_number = int(x["馬番"])
        odds = x[1]
        df = pay_dict[race_id]
        df_tansho = df.loc["単勝"]
        df_tansho["該当馬"] = list(map(int, df_tansho["該当馬"]))
        df_tansho["金"] = list(map(str, df_tansho["金"]))
        invested = 100 
        payback = 0
        if horse_number in df_tansho["該当馬"]:
            idx = df_tansho["該当馬"].index(horse_number)
            tmp = df_tansho["金"][idx]
            tmp = tmp.split(",")
            payback += int("".join(tmp))
        return (payback, invested)
        
    #現在は馬連の計算はできない。
    """
    @staticmethod
    def cal_umaren(x):
        race_id = x.name
        pay_df = pay_dict[race_id]
        df_umaren = pay_df.loc["馬連"]
        umaban_list = x["馬番_list"]
        umaban_list = list(map(int, umaban_list))

        df_umaren["該当馬"] = list(map(int, df_umaren["該当馬"]))
        df_umaren["金"] = list(map(str, df_umaren["金"]))
        length = len(umaban_list)
        invested = ((length * (length - 1)) // 2) * 100  
        payback_sum = 0
        if set(df_umaren["該当馬"]) <= set(umaban_list):
            tmp = df_tansho["金"][0]
            tmp = tmp.split(",")
            tmp = int("".join(tmp))
            payback_sum += tmp
        return (payback_sum, invested)    
    """    
        
    @staticmethod    
    def cal_fukusho(x):
        race_id = x.name
        pay_df = pay_dict[race_id]
        df_fukusho = pay_df.loc["複勝"]
        df_fukusho["該当馬"] = list(map(int, df_fukusho["該当馬"]))
        df_fukusho["金"] = list(map(str, df_fukusho["金"]))
        horse_number = int(x["馬番"])

        invested = 100
        payback_sum = 0
        
        if horse_number in df_fukusho["該当馬"]:
            idx = df_fukusho["該当馬"].index(horse_number)
            tmp = df_fukusho["金"][idx]
            tmp = tmp.split(",")
            tmp = int("".join(tmp))
            payback_sum += tmp

        return (payback_sum, invested)    



class Evaluater():
    drops = ["馬番"]
    X = None
    y = None
    X_train = None
    X_val = None
    X_test = None
    y_train = None
    y_test = None
    y_val = None
    params = {'objective': 'regression',
                     'random_state': 57,
                     'metric': 'l2',
                     'feature_pre_filter': False,
                     'lambda_l1': 0.15883047646498394,
                     'lambda_l2': 9.85103023641964,
                     'num_leaves': 4,
                     'feature_fraction': 0.5,
                     'bagging_fraction': 0.9223910437388337,
                     'bagging_freq': 5,
                     'min_child_samples': 20,
                     'num_iterations': 1000}
    model = lgb.LGBMRegressor(**params)
    
    
    def __init__(self, X, y, odds, test_size=0.3):
        race = list(X.index.unique())
        race_tmp, race_test = train_test_split(race, shuffle=True, test_size=test_size)
        race_train, race_val = train_test_split(race_tmp, shuffle=True, test_size=0.3)
        self.X_train, self.y_train = X[X.index.isin(race_train)], y[X.index.isin(race_train)]
        self.X_test, self.y_test = X[X.index.isin(race_test)], y[X.index.isin(race_test)]
        self.X_val, self.y_val = X[X.index.isin(race_val)], y[X.index.isin(race_val)]
        self.X = X
        self.y = y
        if odds:
            self.drops.append("単勝")
        return 
    
    def predict(self, threshold=0,):
        self.fit()
        pred = self.model.predict(self.X_test.drop(self.drops, axis=1, inplace=False))
        df = pd.DataFrame(pred, index=self.X_test.index, columns=["pred"])
        df["mean"] = df.groupby(df.index)["pred"].transform("mean")
        df["std"] = df.groupby(df.index)["pred"].transform("std")
        df["pred_scaled"] = (df["pred"] - df["mean"]) / df["std"]
        self.pred = pred.copy()
        self.pred[df["pred_scaled"] >= threshold] = 1
        self.pred[df["pred_scaled"] < threshold] = 0
        return self.pred
    
    def fit(self,):
        verbose_eval = -1
        self.model.fit(
            self.X_train.drop(self.drops, axis=1, inplace=False), self.y_train, 
            eval_metric='mean_squared_error', 
            eval_set=[(self.X_val.drop(self.drops, axis=1, inplace=False), self.y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=10, 
                        verbose=False), # early_stopping用コールバック関数
                    lgb.log_evaluation(verbose_eval)] # コマンドライン出力用コールバック関数
                 )

    def importance(self,):
        self.fit()
        importances = pd.DataFrame(
            {
                "features":self.X_train.drop(self.drops, axis=1, inplace=False).columns, "importance":self.model.feature_importances_
            })
            
        importances =  importances.sort_values("importance", ascending=False)
        return importances
    
    def cal(self, threshold=0, fukusho=False, tansho=False, umaren=False,):
            
        if all([not tansho, not fukusho, not umaren]):
            raise ValueError("賭け方を指定してください") 
        if sum([tansho, fukusho, umaren]) != 1:
            raise ValueError("賭け方の指定は一つだけです")
        
        pred = self.predict(threshold,)
        parchaced = self.X_test[pred==1][["馬番","単勝"]]
        invested = 0
        payback_sum = 0

        if tansho:
            paybacks = parchaced.apply(lambda x:Cal.cal_tansho(x)[0], axis=1)
            investeds = parchaced.apply(lambda x:Cal.cal_tansho(x)[1], axis=1)
            if investeds.shape[0] > 0:
                invested += sum(investeds)
                payback_sum += sum(paybacks)
        if fukusho:
            paybacks = parchaced.apply(lambda x:Cal.cal_fukusho(x)[0], axis=1)
            investeds = parchaced.apply(lambda x:Cal.cal_fukusho(x)[1], axis=1)
            if investeds.shape[0] > 0:
                invested += sum(investeds)
                payback_sum += sum(paybacks)
                    
        res = {}
        res["invested"] = invested
        res["payback_sum"] = payback_sum
        res["div"] = payback_sum - invested
        if invested == 0:
            res["kaishuuritu"] = 1 * 100
        else:
            res["kaishuuritu"] = (payback_sum / invested)*100
            
        return res
    
    def visualize(self, bins=20, fukusho=False, tansho=False, umaren=False,):
        thresholds = []
        kaishuuritu = []
        divs = []
        investeds = []
        res = {}
        for i in reversed(range(bins + 2)):
            threshold = -3 + (i / bins)*6
            thresholds.append(threshold)
            info = self.cal(threshold=threshold, tansho=tansho, fukusho=fukusho, umaren=umaren,)
            divs.append(info["div"])
            investeds.append(info["invested"])
            kaishuuritu.append(info["kaishuuritu"])
        res["invested"] = np.array(investeds)
        res["kaishuuritu"] = np.array(kaishuuritu)
        res["div"] = np.array(divs)
        res["threshold"] = np.array(thresholds)
        res["bet_percent"] = res["invested"]*100 / np.sum(res["invested"])
        for i in range(res["bet_percent"].shape[0] - 1):
            res["bet_percent"][i + 1] += res["bet_percent"][i]
        return res   
   





def get_simulation_dict(n_bins=20, features={}, test_size=0.3, feature_combinations=[]):
    data = pd.read_pickle(data_path)
    X, y = data.drop(["着順"], axis=1), -data["着順"]
   
    odds = False  
    use_features = ["馬番","単勝", "セ"]

    for feature, is_used in features.items():
        if is_used:
            add_feature = MyForm.param_to_feature[feature]
            if add_feature == "単勝":
                odds = True
                continue
            use_features.append(add_feature)

    X = X[use_features]
    if len(feature_combinations) > 0:
        synthesis_features = []
        for feature_combination in feature_combinations:
            synthesis_features.append((feature_combination, pm.MUL))
        sr = pm.SynthesisReactor(synthesis_features)
        X = sr.fit_transform(X)

    
    ev = Evaluater(X, y, odds=odds, test_size=test_size)
    output = ev.visualize(tansho=True, bins=n_bins,)
    return output, X.columns




