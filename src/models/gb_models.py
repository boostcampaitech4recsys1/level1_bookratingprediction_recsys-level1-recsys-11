import tqdm

import numpy as np
import pandas as pd
import os

from scipy.sparse import csr_matrix, linalg
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier, Pool
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

    # def train(self, fold_num):
    #     # model: type, optimizer: torch.optim, train_dataloader: DataLoader, criterion: torch.nn, device: str, log_interval: int=100

    #         self.model.train()
    #         total_loss = 0
    #         tk0 = tqdm.tqdm(self.train_dataloader, smoothing=0, mininterval=1.0)
    #         for i, (fields, target) in enumerate(tk0):
    #             self.model.zero_grad()
    #             fields, target = fields.to(self.device), target.to(self.device)

    #             y = self.model(fields)
    #             loss = self.criterion(y, target.float())

    #             loss.backward()
    #             self.optimizer.step()
    #             total_loss += loss.item()
    #             if (i + 1) % self.log_interval == 0:
    #                 tk0.set_postfix(loss=total_loss / self.log_interval)
    #                 total_loss = 0

    #         rmse_score = self.predict_train()
    #         early_stopping(rmse_score, self.model)  

    #         if early_stopping.early_stop:
    #             print("Early stopping")
    #             break

    #     formatted_user_num = format(self.args.USER_NUM, '02')
    #     formatted_book_num = format(self.args.BOOK_NUM, '02')
    #     ppath = os.path.join(self.args.SAVE_PATH,
    #         self.args.MODEL,
    #         f"u{formatted_user_num}_b{formatted_book_num}",
    #         f"fold{fold_num}",
    #         'checkpoint.pt')
    #     self.model.load_state_dict(torch.load(ppath))
    #     rmse_score = self.predict_train()
    #     print(f"u{formatted_user_num}_b{formatted_book_num}, validation rmse: {rmse_score}")
    #     print('\n')



    # def predict_train(self):
    #     self.model.eval()
    #     targets, predicts = list(), list()
    #     with torch.no_grad():
    #         for fields, target in tqdm.tqdm(self.valid_dataloader, smoothing=0, mininterval=1.0):
    #             fields, target = fields.to(self.device), target.to(self.device)
    #             y = self.model(fields)
    #             targets.extend(target.tolist())
    #             predicts.extend(y.tolist())
    #     return rmse(targets, predicts)


    # def predict(self, dataloader):
    #     self.model.eval()
    #     predicts = list()
    #     with torch.no_grad():
    #         for fields in tqdm.tqdm(dataloader, smoothing=0, mininterval=1.0):
    #             fields = fields[0].to(self.device)
    #             y = self.model(fields)
    #             predicts.extend(y.tolist())
    #     return predicts