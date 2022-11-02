import tqdm

import numpy as np
import pandas as pd
import os

from scipy.sparse import csr_matrix, linalg
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier, Pool

from ._models import rmse, acc, confusion_mat

import warnings


# from ._models import _NeuralCollaborativeFiltering, _WideAndDeepModel, _DeepCrossNetworkModel
from ._models import rmse, RMSELoss

class XGBoostModel:

    def __init__(self, args, data, cf=True):
        self.args = args

        self.train_data = (pd.get_dummies(
            data['train_dataloader'][0].drop(columns=['user_id', 'isbn', 'book_author'], axis=1),
            columns=['location_country', 'age', 'year_of_publication', 'publisher', 'category']
        ), data['train_dataloader'][1])

        self.valid_data = (pd.get_dummies(
            data['valid_dataloader'][0].drop(columns=['user_id', 'isbn', 'book_author'], axis=1),
            columns=['location_country', 'age', 'year_of_publication', 'publisher', 'category']
        ), data['valid_dataloader'][1])
        self.max_depth = args.XGB_MAX_DEPTH

        ## 클래시파이어로 변환하는 과정 및 로스 교체 코드
        self.cf = cf
               ## 리그레션 일 시 클래시파이어 일시 달라짐
        if cf:
            self.learning_rate = args.CF_LR
            self.model = XGBClassifier(learning_rate = self.learning_rate, max_depth = self.max_depth)
        else:
            self.learning_rate = args.RR_LR
            self.model = XGBRegressor(learning_rate = self.learning_rate, max_depth = self.max_depth)



    def train(self, fold_num):
        X, y = self.train_data
        print(f'XGBoost training... ', end='', flush=True)
        self.model.fit(X, y)
        print(f'done.')

        formatted_user_num = format(self.args.USER_NUM, '02')
        formatted_book_num = format(self.args.BOOK_NUM, '02')
        ppath = os.path.join(self.args.SAVE_PATH,
            self.args.CF_MODEL, '+', self.args.RR_MODEL, ## 클래시파이어 수정 부분
            f"u{formatted_user_num}_b{formatted_book_num}",
            f"fold{fold_num}",
            'xgb_model.model')
        rmse_score = self.predict_train()
        print(f"u{formatted_user_num}_b{formatted_book_num}, validation rmse: {rmse_score}")
        print('\n')


    def predict_train(self):
        X, targets = self.train_data
        predicts = self.model.predict(X)

        # 클래시파이어 부분
        if self.cf:
            # print(np.argmax(predicts, axis=1), targets)
            t = np.get_printoptions()
            np.set_printoptions(precision=2)

            print('[confusion matrix] row: real, col: pred\n', confusion_mat(targets, predicts) * 100)
            print('[classification acc]:', f'{acc(targets, predicts) * 100:.3f}%')
            np.set_printoptions(precision=t['precision'])

            return rmse(targets, predicts)

        if self.args.ZEROONE:
            return rmse([t * 10.0 for t in targets], [p * 10.0 for p in predicts])
        else:
            return rmse(targets, predicts)


    def predict(self, dataloader):
        dataloader = (pd.get_dummies(
            dataloader[0].drop(columns=['user_id', 'isbn', 'book_author'], axis=1),
            columns=['location_country', 'age', 'year_of_publication', 'publisher', 'category']
        ))
        predicts = self.model.predict(dataloader[0])
        return predicts


class LightGBMModel:

    def __init__(self, args, data, cf):
        self.args = args
        print(data['train_dataloader'][0])
        self.train_data = (pd.get_dummies(
            data['train_dataloader'][0].drop(columns=['user_id', 'isbn', 'book_author'], axis=1),
            columns=['location_country', 'age', 'year_of_publication', 'publisher', 'category']
        ), data['train_dataloader'][1])

        self.valid_data = (pd.get_dummies(
            data['valid_dataloader'][0].drop(columns=['user_id', 'isbn', 'book_author'], axis=1),
            columns=['location_country', 'age', 'year_of_publication', 'publisher', 'category']
        ), data['valid_dataloader'][1])

        ## 클래시파이어로 변환하는 과정 및 로스 교체 코드
        self.cf = cf
        ## 리그레션 일 시 클래시파이어 일시 달라짐
        if cf:
            self.learning_rate = args.CF_LR
            self.model = LGBMClassifier(learning_rate = self.learning_rate)
        else:
            self.learning_rate = args.RR_LR
            self.model = LGBMRegressor(learning_rate = self.learning_rate)


    def train(self, fold_num):
        X, y = self.train_data
        print(f'LightGBM training... ', end='', flush=True)
        print(X)
        print(y)
        self.model.fit(X, y)
        print(f'done.')

        predicts = self.model.predict(X)
        t = np.get_printoptions()
        np.set_printoptions(precision=2)

        print('[confusion matrix] row: real, col: pred\n', confusion_mat(y, predicts) * 100)
        print('[classification acc]:', f'{acc(y, predicts) * 100:.3f}%')
        np.set_printoptions(precision=t['precision'])

        formatted_user_num = format(self.args.USER_NUM, '02')
        formatted_book_num = format(self.args.BOOK_NUM, '02')
        ppath = os.path.join(self.args.SAVE_PATH,
            self.args.CF_MODEL, '+', self.args.RR_MODEL, ## 클래시파이어 수정 부분
            f"u{formatted_user_num}_b{formatted_book_num}",
            f"fold{fold_num}")
        rmse_score = self.predict_train()
        print(f"u{formatted_user_num}_b{formatted_book_num}, validation rmse: {rmse_score}")
        print('\n')


    def predict_train(self):
        X, targets = self.train_data
        predicts = self.model.predict(X)

        # 클래시파이어 부분
        if self.cf:
            # print(np.argmax(predicts, axis=1), targets)
            t = np.get_printoptions()
            np.set_printoptions(precision=2)

            print('[confusion matrix] row: real, col: pred\n', confusion_mat(targets, predicts) * 100)
            print('[classification acc]:', f'{acc(targets, predicts) * 100:.3f}%')
            np.set_printoptions(precision=t['precision'])

            return rmse(targets, predicts)

        if self.args.ZEROONE:
            return rmse([t * 10.0 for t in targets], [p * 10.0 for p in predicts])
        else:
            return rmse(targets, predicts)


    def predict(self, dataloader):
        dataloader = (pd.get_dummies(
            dataloader[0].drop(columns=['user_id', 'isbn', 'book_author'], axis=1),
            columns=['location_country', 'age', 'year_of_publication', 'publisher', 'category']
        ))
        predicts = self.model.predict(dataloader[0])
        return predicts


class CatBoostModel:

    def __init__(self, args, data, cf):
        self.args = args

        self.train_data = data['train_dataloader']
        self.valid_data = data['valid_dataloader']

        ## 클래시파이어로 변환하는 과정 및 로스 교체 코드
        self.cf = cf
        ## 리그레션 일 시 클래시파이어 일시 달라짐
        if cf:
            self.learning_rate = args.CF_LR
            self.model = CatBoostClassifier(learning_rate = self.learning_rate, verbose=200)
        else:
            self.learning_rate = args.RR_LR
            self.model = CatBoostRegressor(learning_rate = self.learning_rate, verbose=200)

    def train(self, fold_num):
        X, y = self.train_data
        print(f'CatBoost training... ', end='', flush=True)
        self.model.fit(X, y)
        print(f'done.')
        # 클래시파이어 부분
        formatted_user_num = format(self.args.USER_NUM, '02')
        formatted_book_num = format(self.args.BOOK_NUM, '02')
        ppath = os.path.join(self.args.SAVE_PATH,
            self.args.CF_MODEL, '+', self.args.RR_MODEL, ## 클래시파이어 수정 부분
            f"u{formatted_user_num}_b{formatted_book_num}",
            f"fold{fold_num}")
        rmse_score = self.predict_train()
        print(f"u{formatted_user_num}_b{formatted_book_num}, validation rmse: {rmse_score}")
        print('\n')


    def predict_train(self):
        X, targets = self.train_data
        predicts = self.model.predict(X)

        # 클래시파이어 부분
        if self.cf:
            # print(np.argmax(predicts, axis=1), targets)
            t = np.get_printoptions()
            np.set_printoptions(precision=2)

            print('[confusion matrix] row: real, col: pred\n', confusion_mat(targets, predicts) * 100)
            print('[classification acc]:', f'{acc(targets, predicts) * 100:.3f}%')
            np.set_printoptions(precision=t['precision'])

            return rmse(targets, predicts)

        if self.args.ZEROONE:
            return rmse([t * 10.0 for t in targets], [p * 10.0 for p in predicts])
        else:
            return rmse(targets, predicts)


    def predict(self, dataloader):
        predicts = self.model.predict(dataloader[0])
        return predicts