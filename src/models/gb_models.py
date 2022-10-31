import tqdm

import numpy as np
import pandas as pd
import os

from scipy.sparse import csr_matrix, linalg
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
# from catboost import CatBoostRegressor, CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score

import warnings


# from ._models import _NeuralCollaborativeFiltering, _WideAndDeepModel, _DeepCrossNetworkModel
from ._models import rmse, RMSELoss

class XGBoostModel:

    def __init__(self, args, data):
        self.args = args

        self.train_data = data['train_dataloader']
        self.valid_data = data['valid_dataloader']

        self.learning_rate = args.LR
        self.max_depth = args.XGB_MAX_DEPTH

        warnings.filterwarnings(action='ignore')

        if args.XGB_RR_CL.lower() == 'rr':
            self.model = XGBRegressor(learning_rate = self.learning_rate, max_depth = self.max_depth)
        elif args.XGB_RR_CL.lower() == 'cl':
            self.model = XGBClassifier(learning_rate = self.learning_rate, max_depth = self.max_depth)
        else:
            raise Execption('XGB 모드 선택 오류: rr, cl 중에 선택 가능합니다.')

    def train(self, fold_num):
        X, y = self.train_data
        print(f'XGBoost {self.args.XGB_RR_CL.lower()} training... ', end='', flush=True)
        self.model.fit(X, y)
        print(f'done.')
        formatted_user_num = format(self.args.USER_NUM, '02')
        formatted_book_num = format(self.args.BOOK_NUM, '02')
        ppath = os.path.join(self.args.SAVE_PATH,
            self.args.MODEL,
            f"u{formatted_user_num}_b{formatted_book_num}",
            f"fold{fold_num}",
            'xgb_model.model')
        rmse_score = self.predict_train()
        print(f"u{formatted_user_num}_b{formatted_book_num}, validation rmse: {rmse_score}")
        print('\n')


    def predict_train(self):
        X, y = self.train_data
        y_hat = self.model.predict(X)
        return rmse(y_hat, y)


    def predict(self, dataloader):
        predicts = self.model.predict(dataloader[0])
        return predicts


class LightGBMModel:

    def __init__(self, args, data):
        self.args = args

        self.train_data = data['train_dataloader']
        self.valid_data = data['valid_dataloader']

        self.learning_rate = args.LR
        # self.max_depth = args.XGB_MAX_DEPTH

        if args.LGBM_RR_CL.lower() == 'rr':
            self.model = LGBMRegressor(learning_rate = self.learning_rate)
        elif args.LGBM_RR_CL.lower() == 'cl':
            self.model = LGBMClassifier(learning_rate = self.learning_rate)
        else:
            raise Execption('XGB 모드 선택 오류: rr, cl 중에 선택 가능합니다.')

    def train(self, fold_num):
        X, y = self.train_data
        print(f'LightGBM {self.args.LGBM_RR_CL.lower()} training... ', end='', flush=True)
        self.model.fit(X, y)
        print(f'done.')
        formatted_user_num = format(self.args.USER_NUM, '02')
        formatted_book_num = format(self.args.BOOK_NUM, '02')
        ppath = os.path.join(self.args.SAVE_PATH,
            self.args.MODEL,
            f"u{formatted_user_num}_b{formatted_book_num}",
            f"fold{fold_num}",
            'xgb_model.model')
        rmse_score = self.predict_train()
        print(f"u{formatted_user_num}_b{formatted_book_num}, validation rmse: {rmse_score}")
        print('\n')


    def predict_train(self):
        X, y = self.train_data
        y_hat = self.model.predict(X)
        return rmse(y_hat, y)


    def predict(self, dataloader):
        predicts = self.model.predict(dataloader[0])
        return predicts


class CatBoostModel:

    def __init__(self, args, data):
        self.args = args

        self.train_data = data['train_dataloader']
        self.valid_data = data['valid_dataloader']

        self.learning_rate = args.LR
        # self.max_depth = args.XGB_MAX_DEPTH

        if args.CATB_RR_CL.lower() == 'rr':
            self.model = CatBoostRegressor(learning_rate = self.learning_rate, verbose=200)
        elif args.CATB_RR_CL.lower() == 'cl':
            self.model = CatBoostClassifier(learning_rate = self.learning_rate, verbose=200)
        else:
            raise Execption('XGB 모드 선택 오류: rr, cl 중에 선택 가능합니다.')

    def train(self, fold_num):
        X, y = self.train_data
        print(f'CATB {self.args.CATB_RR_CL.lower()} training... ', flush=True)
        self.model.fit(X, y)
        print(f'done.')
        formatted_user_num = format(self.args.USER_NUM, '02')
        formatted_book_num = format(self.args.BOOK_NUM, '02')
        ppath = os.path.join(self.args.SAVE_PATH,
            self.args.MODEL,
            f"u{formatted_user_num}_b{formatted_book_num}",
            f"fold{fold_num}",
            'xgb_model.model')
        rmse_score = self.predict_train()
        print(f"u{formatted_user_num}_b{formatted_book_num}, validation rmse: {rmse_score}")
        print('\n')


    def predict_train(self):
        X, y = self.train_data
        y_hat = self.model.predict(X)
        return rmse(y_hat, y)


    def predict(self, dataloader):
        predicts = self.model.predict(dataloader[0])
        return predicts