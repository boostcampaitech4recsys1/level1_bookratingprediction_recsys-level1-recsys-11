import os
import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from pathlib import Path

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class EarlyStopping:
    def __init__(self, args, fold_num, verbose=False, delta=0):
        self.args = args
        self.fold_num = fold_num
        self.patience = args.PATIENCE
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(
                f"Validation rmse decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        formatted_user_num = format(self.args.USER_NUM, '02')
        formatted_book_num = format(self.args.BOOK_NUM, '02')
        ppath = Path(
            os.path.join(
                self.args.SAVE_PATH,
                self.args.MODEL,
                f"u{formatted_user_num}_b{formatted_book_num}",
                f"fold{self.fold_num}",
                'checkpoint.pt'
            )
        )
        ppath.parent.mkdir(parents=True, exist_ok=True)

        torch.save(model.state_dict(), str(ppath))
        self.val_loss_min = val_loss