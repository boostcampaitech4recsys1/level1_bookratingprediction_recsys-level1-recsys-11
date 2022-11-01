############### 
# 이 main2.py 는 클래시파이어 역할을 하는 모델, 리그레서 역할을 하는 모델로 나뉩니다.
# 따라서 역할에 맞는 배치가 필요합니다.
# 리그레서만 가능한 모델이 클래시파이어로 arg 할당되면 돌릴 수 없습니다.
# 캐스케이드 앙상블을 합니다.
# ex) python main2.py --USER_NUM 4 --BOOK_NUM 5 --CF_MODEL NCF --RR_MODEL FM --CF_LR 0.0001 --RR_LR 0.00001 --CF_BATCH_SIZE 64 --RR_BATCH_SIZE 16

import enum
import time
import argparse
import pandas as pd
import numpy as np

from src import seed_everything

from src.data import context_data_load, context_data_split, context_data_loader
from src.data import dl_data_load, dl_data_split, dl_data_loader
from src.data import image_data_load, image_data_split, image_data_loader
from src.data import text_data_load, text_data_split, text_data_loader

from src import FactorizationMachineModel
from src import NeuralCollaborativeFiltering

from src import XGBoostModel, LightGBMModel, CatBoostModel

from sklearn.model_selection import StratifiedKFold

def rmse(real: list, predict: list) -> float:
    pred = np.array(predict)
    return np.sqrt(np.mean((real-pred) ** 2))


def main(args):
    seed_everything(args.SEED)

    ######################## DATA LOAD
    print(f'--------------- {args.CF_MODEL} Load Data ---------------')
    data = dl_data_load(args)
    
    ranges = args.RANGE.split(',')
    data['ranges'] = ranges

    ###########어떤 로스, 제로원인지 알려주기용
    if args.ZEROONE:
        print('zeroone')
    else:
        print('no zeroone')

    if args.LOSS == 'sl1':
        print('with sl1 loss beta', args.BETA)
    elif args.LOSS == 'rmse':
        print('with rmse loss')


    if args.VALID == 'random':
        ######################## Train/Valid Split, Load
        print(f'--------------- CF:{args.CF_MODEL} ~ RR:{args.RR_MODEL} Train/Valid Split ---------------')
        # 최초 스플릿은 그대로.
        data = dl_data_split(args, data)
        
        # 클래시피케이션 용 데이터로더 추가
        if args.CF_MODEL in ('XGB', 'LGBM', 'CATB'):
            print('?')
            data_cf = dl_data_loader(args, data, cf=True, is_boosting=True)
        else:
            data_cf = dl_data_loader(args, data, cf=True)

        # 리그레션용 데이터로더 추가
        datas_rr = []
        for r in ranges:
            if args.RR_MODEL in ('XGB', 'LGBM', 'CATB'):
                data_rr = dl_data_loader(args, data, range_str=r, is_boosting=True)
            else:
                data_rr = dl_data_loader(args, data, range_str=r)
            datas_rr.append(data_rr)

        # 학습 후 벨리드 셋 평가용
        data_v = dl_data_loader(args, data, last_valid=True)

        ######################## Model
        print(f'--------------- INIT CF:{args.CF_MODEL} ---------------')
        
        if args.CF_MODEL == 'FM':
            model_cf = FactorizationMachineModel(args, data_cf, cf=True) ## cf True 이면 RANGE 로 클래스 분류    
        elif args.CF_MODEL == 'NCF':
            model_cf = NeuralCollaborativeFiltering(args, data_cf, cf=True)
        elif args.CF_MODEL=='XGB':
            model_cf = XGBoostModel(args, data_cf, cf=True)
        elif args.CF_MODEL=='LGBM':
            model_cf = LightGBMModel(args, data_cf, cf=True)
        elif args.CF_MODEL=='CATB':
            model_cf = CatBoostModel(args, data_cf, cf=True)
        else:
            pass    

        print(f'--------------- INIT RR:{args.RR_MODEL} ---------------')

        models_rr = []
        if args.RR_MODEL == 'FM':
            for data_rr in datas_rr:
                models_rr.append(FactorizationMachineModel(args, data_rr))
        elif args.RR_MODEL == 'NCF':
            for data_rr in datas_rr:
                models_rr.append(NeuralCollaborativeFiltering(args, data_rr))
        elif args.RR_MODEL == 'XGB':
            for data_rr in datas_rr:
                models_rr.append(XGBoostModel(args, data_rr))
        elif args.RR_MODEL == 'LGBM':
            for data_rr in datas_rr:
                models_rr.append(LightGBMModel(args, data_rr))
        elif args.RR_MODEL == 'CATB':
            for data_rr in datas_rr:
                models_rr.append(CatBoostModel(args, data_rr))

        # 학습 후 벨리드 셋 평가용
        data_v = dl_data_loader(args, data, last_valid=True)

        ######################## TRAIN
        print(f'--------------- CF:{args.CF_MODEL} TRAINING ---------------')
        model_cf.train(fold_num = 0)

        for i, model_rr in enumerate(models_rr):
            print(f'--------------- RR:{args.RR_MODEL} [{i+1}/{len(models_rr)}] TRAINING ---------------')
            model_rr.train(fold_num = 0)
        
        ######################## VALDIATION rmse 
        print(f'--------------- GET VALIDATION SCORE ---------------')
        v_predict_cf = model_cf.predict(data_v['valid_dataloader'])
        print(list(v_predict_cf).count(0))
        print(list(v_predict_cf).count(1))
        print(list(v_predict_cf).count(2))

        range_nums = list(range(len(ranges)))
        v_predicts_rr = []

        for model_rr in models_rr:
            v_predicts_rr.append(model_rr.predict(data_v['valid_dataloader']))

        predicts = np.zeros_like(v_predict_cf, dtype=np.float64)
        
        for i, v_predict_rr in enumerate(v_predicts_rr):
            print(np.where(v_predict_cf == i)[0])
            indices = np.where(v_predict_cf == i)[0]
            for idx in indices:
                predicts[idx] = v_predict_rr[idx]

        if args.ZEROONE: # 0. ~ 1 스케일링 시
            valid_rmse = rmse(data_v['y_valid'], [p * 10.0 for p in predicts])
        else:
            valid_rmse = rmse(data_v['y_valid'], predicts)
        print('Final validation score:', valid_rmse)


        ######################## INFERENCE
        print(f'--------------- CF:{args.CF_MODEL} ~ RR:{args.RR_MODEL} PREDICT ---------------')
        t_predict_cf = model_cf.predict(data_v['test_dataloader'])
        print(list(t_predict_cf).count(0))
        print(list(t_predict_cf).count(1))
        print(list(t_predict_cf).count(2))

        range_nums = list(range(len(ranges)))
        t_predicts_rr = []

        for model_rr in models_rr:
            t_predicts_rr.append(model_rr.predict(data_v['test_dataloader']))

        predicts = np.zeros_like(t_predict_cf, dtype=np.float64)
        
        for i, t_predict_rr in enumerate(t_predicts_rr):
            print(np.where(t_predict_cf == i)[0])
            indices = np.where(t_predict_cf == i)[0]
            for idx in indices:
                predicts[idx] = t_predict_rr[idx]
        print(predicts)

        ######################## SAVE PREDICT
        print(f'--------------- SAVE CF:{args.CF_MODEL} ~ RR:{args.RR_MODEL} PREDICT ---------------')
        submission = pd.read_csv(args.DATA_PATH + 'ratings/sample_submission.csv')
        if args.ZEROONE: # 0. ~ 1 스케일링 시
            submission['rating'] = [p * 10.0 for p in predicts]
        else:
            submission['rating'] = predicts
        

    elif args.VALID == 'kfold':
        skf = StratifiedKFold(n_splits = args.N_SPLITS, shuffle = True)
        length = len(data['test'])
        kfold_predicts = np.zeros((args.N_SPLITS, length))
        rmse_array = np.zeros(args.N_SPLITS)


        for idx, (train_index, valid_index) in enumerate(skf.split(
                                            data['train'].drop(['rating'], axis = 1),
                                            data['train']['rating']
                                            )):
            
            data['X_train']= data['train'].drop(['rating'], axis = 1).iloc[train_index]
            data['y_train'] = data['train']['rating'].iloc[train_index]
            data['X_valid']= data['train'].drop(['rating'], axis = 1).iloc[valid_index]
            data['y_valid'] = data['train']['rating'].iloc[valid_index]

            # 클래시피케이션 용 데이터로더 추가
            data_cf = dl_data_loader(args, data, cf=True)

            # 리그레션용 데이터로더 추가
            datas_rr = []
            for r in ranges:
                data_rr = dl_data_loader(args, data, range_str=r)
                datas_rr.append(data_rr)

            ######################## Model
            print(f'--------------- FOLD-{idx}, INIT CF:{args.CF_MODEL} ---------------')
            if args.CF_MODEL == 'FM':
                model_cf = FactorizationMachineModel(args, data_cf, cf=True) ## cf True 이면 RANGE 로 클래스 분류    
            elif args.CF_MODEL == 'NCF':
                model_cf = NeuralCollaborativeFiltering(args, data_cf, cf=True)         
            
            print(f'--------------- FOLD-{idx}, INIT RR:{args.RR_MODEL} ---------------')
            models_rr = []
            if args.RR_MODEL == 'FM':
                for data_rr in datas_rr:
                    models_rr.append(FactorizationMachineModel(args, data_rr))
            elif args.RR_MODEL == 'NCF':
                for data_rr in datas_rr:
                    models_rr.append(NeuralCollaborativeFiltering(args, data_rr))

            # 학습 후 벨리드 셋 평가용
            data_v = dl_data_loader(args, data, last_valid=True)

            ######################## TRAIN
            print(f'--------------- FOLD-{idx}, CF:{args.CF_MODEL} TRAINING ---------------')
            model_cf.train(fold_num = idx)
            for i, model_rr in enumerate(models_rr):
                print(f'--------------- FOLD-{idx}, RR:{args.RR_MODEL} [{i+1}/{len(models_rr)}] TRAINING ---------------')
                model_rr.train(fold_num = idx)
            
            ######################## VALDIATION rmse 
            print(f'--------------- FOLD-{idx}, GET VALIDATION SCORE ---------------')
            v_predict_cf = model_cf.predict(data_v['valid_dataloader'])
            print(list(v_predict_cf).count(0))
            print(list(v_predict_cf).count(1))
            print(list(v_predict_cf).count(2))

            range_nums = list(range(len(ranges)))
            v_predicts_rr = []

            for model_rr in models_rr:
                v_predicts_rr.append(model_rr.predict(data_v['valid_dataloader']))

            predicts = np.zeros_like(v_predict_cf, dtype=np.float64)
            
            for i, v_predict_rr in enumerate(v_predicts_rr):
                print(np.where(v_predict_cf == i)[0])
                indices = np.where(v_predict_cf == i)[0]
                for idx_ in indices:
                    predicts[idx_] = v_predict_rr[idx_]

            if args.ZEROONE: # 0. ~ 1 스케일링 시
                valid_rmse = rmse(data_v['y_valid'], [p * 10.0 for p in predicts])
            else:
                valid_rmse = rmse(data_v['y_valid'], predicts)
            print('FOLD-{idx} Final validation score:', valid_rmse)
            rmse_array[idx] = valid_rmse

            ######################## INFERENCE
            print(f'--------------- FOLD-{idx}, {args.CF_MODEL} ~ {args.RR_MODEL} PREDICT ---------------')

            t_predict_cf = model_cf.predict(data_v['test_dataloader'])
            print(list(t_predict_cf).count(0))
            print(list(t_predict_cf).count(1))
            print(list(t_predict_cf).count(2))

            range_nums = list(range(len(ranges)))
            t_predicts_rr = []

            for model_rr in models_rr:
                t_predicts_rr.append(model_rr.predict(data_v['test_dataloader']))

            predicts = np.zeros_like(t_predict_cf, dtype=np.float64)
            
            for i, t_predict_rr in enumerate(t_predicts_rr):
                print(np.where(t_predict_cf == i)[0])
                indices = np.where(t_predict_cf == i)[0]
                for idx_ in indices:
                    predicts[idx_] = t_predict_rr[idx_]
            kfold_predicts[idx] = predicts


        print(f'--------------- FOLD-{idx}, SAVE {args.CF_MODEL} PREDICT ---------------')
        predicts = np.mean(kfold_predicts, axis = 0).tolist()
        submission = pd.read_csv(args.DATA_PATH + 'ratings/sample_submission.csv')
        print(f"[5-FOLD VALIDATION MEAN RMSE SCORE]: {rmse_array.mean()}")

        ######################## SAVE PREDICT
        if args.ZEROONE: # 0. ~ 1 스케일링 시
            submission['rating'] = submission['rating'] = [p * 10 for p in predicts]
        else:
            submission['rating'] = predicts


    now = time.localtime()
    now_date = time.strftime('%Y%m%d', now)
    now_hour = time.strftime('%X', now)
    save_time = now_date + '_' + now_hour.replace(':', '')
    print(f"[SUBMISSION NAME] {save_time}_{args.CF_MODEL}_{args.RR_MODEL} @@@@")
    if args.ROUND: # 라운드 된 것 안된 것 둘다 저장하기.
        submission_r = submission.copy()
        submission_r['rating'] = submission_r['rating'].apply(np.round)
        submission_r.to_csv('/opt/ml/data/submit/{}_{}_{}r.csv'.format(save_time, args.CF_MODEL, args.RR_MODEL),index=False)
    submission.to_csv('/opt/ml/data/submit/{}_{}_{}.csv'.format(save_time, args.CF_MODEL, args.RR_MODEL), index=False)


if __name__ == "__main__":

    ######################## BASIC ENVIRONMENT SETUP
    parser = argparse.ArgumentParser(description='parser')
    arg = parser.add_argument

    ############### BASIC OPTION
    arg('--DATA_PATH', type=str, default='/opt/ml/data/', help = 'Data path를 설정할 수 있습니다.')
    arg('--SAVE_PATH', type = str, default = '/opt/ml/weights/', help = "학습된 모델들이 저장되는 path입니다.")
    arg('--USER_NUM', type = int, help = "user data preprocessed number `1 ~ 9`")
    arg('--BOOK_NUM', type = int, help = "book data preprocessed number `1 ~ 24`")


    ############### CF, RR 설정
    arg('--CF_MODEL', type=str, choices=['FM', 'NCF', 'XGB', 'LGBM', 'CATB'], help='앞단에서 분포 뿌리기로 클래시피케이션할 모델 선택할 수 있습니다.')
    arg('--RR_MODEL', type=str, choices=['FM', 'NCF', 'XGB', 'LGBM', 'CATB'], help='뒷단에서 점수를 예측할 모델 선택합니다.')

    arg('--CF_BATCH_SIZE', type=int, default=64, help='Batch size를 조정할 수 있습니다.')
    arg('--CF_LR', type=float, default=1e-4, help='Learning Rate를 조정할 수 있습니다.')

    arg('--RR_BATCH_SIZE', type=int, default=32, help='Batch size를 조정할 수 있습니다.')
    arg('--RR_LR', type=float, default=1e-3, help='Learning Rate를 조정할 수 있습니다.')

    ############### 범위 설정
    arg('--RANGE', type=str, default='04,57,89', help='뒷단 모델의 평점 분포의 범위를 정할 수 있습니다. 0~9 사이로 합니다')


    arg('--DATA_SHUFFLE', type=bool, default=True, help='데이터 셔플 여부를 조정할 수 있습니다.')
    arg('--TEST_SIZE', type=float, default=0.2, help='Train/Valid split 비율을 조정할 수 있습니다.')
    arg('--SEED', type=int, default=42, help='seed 값을 조정할 수 있습니다.')
    arg('--VALID', type = str, default = 'kfold', help = "kfold, random")
    arg('--N_SPLITS', type = int, default = 5)
    
    ############### TRAINING OPTION
    arg('--BATCH_SIZE', type=int, default=64, help='Batch size를 조정할 수 있습니다.')
    arg('--EPOCHS', type=int, default=50, help='Epoch 수를 조정할 수 있습니다.')
    arg('--LR', type=float, default=1e-4, help='Learning Rate를 조정할 수 있습니다.')
    arg('--WEIGHT_DECAY', type=float, default=1e-5, help='Adam optimizer에서 정규화에 사용하는 값을 조정할 수 있습니다.')
    arg('--PATIENCE', type = int, default = 3, help = 'Early Stop patience')
    arg('--ZEROONE', type=bool, default=False, help = '0. ~ 1 스케일링 합니다.')
    arg('--ROUND', type=bool, default=False, help = '점수 반올림 진행합니다.')

    ############### Loss Func
    arg('--LOSS', type=str, default='rmse', help='rmse, sl1, huber')
    arg('--BETA', type=float, default=1.0, help='smooth l1, hubor 에서 베타, 델타 지정합니다.(0 ~ 1)')

    ############### GPU
    arg('--DEVICE', type=str, default='cuda', choices=['cuda', 'cpu'], help='학습에 사용할 Device를 조정할 수 있습니다.')

    ############### FM
    arg('--FM_EMBED_DIM', type=int, default=2, help='FM에서 embedding시킬 차원을 조정할 수 있습니다.')

    ############### NCF
    arg('--NCF_EMBED_DIM', type=int, default=16, help='NCF에서 embedding시킬 차원을 조정할 수 있습니다.')
    arg('--NCF_MLP_DIMS', type=list, default=(16, 16), help='NCF에서 MLP Network의 차원을 조정할 수 있습니다.')
    arg('--NCF_DROPOUT', type=float, default=0.2, help='NCF에서 Dropout rate를 조정할 수 있습니다.')

    ############### XGB
    arg('--XGB_MAX_DEPTH', type=int, default=5, help='XGB에서 트리 깊이 지정하며 깊을수록 복잡한 모델이 됩니다.')

    args = parser.parse_args()
    main(args)