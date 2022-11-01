import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

def text_process(vector):
    while '\n' in vector:
        vector = vector.replace('\n', '')
    while '[' in vector:
        vector = vector.replace('[', '')
    while ']' in vector:
        vector = vector.replace(']', '')
    result = np.array(list(map(float, vector.split())))
    return result
    

def process2npy(save_dir, mode, df, case, column_name):
    # df[column_name] = df[column_name].apply(text_process)
    print(f"[MODE: {mode}, COLUMN_NAME: {column_name}]")
    process_list = df[column_name].tolist()
    result_list = []
    for item in tqdm(process_list):
        result_list.append(text_process(item))
    df[f"new_{column_name}"] = result_list
    result_df = df[[case, f"new_{column_name}"]]
    vector = np.concatenate([
                        result_df[case].values.reshape(1, -1),
                        result_df[f"new_{column_name}"].values.reshape(1, -1)
    ])
    ppath = Path(os.path.join(save_dir, f"{column_name}.npy"))
    ppath.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(ppath), vector)
    


def main(mode):
    root_dir = "/opt/ml/data/"
    save_dir = f"/opt/ml/data/embedding/{mode}"
    embedding_df = pd.read_csv(os.path.join(root_dir, f"{mode}_embedding.csv"))
    # case_name: user_id / isbn
    # column_name: user_summary_merge_vector, item_summary_vector, book_title_vector, image_embed

    process2npy(save_dir, mode, embedding_df, 'user_id', 'user_summary_merge_vector')
    process2npy(save_dir, mode, embedding_df, 'isbn', 'item_summary_vector')
    process2npy(save_dir, mode, embedding_df, 'isbn', 'book_title_vector')
    process2npy(save_dir, mode, embedding_df, 'isbn', 'image_embed')


if __name__ == '__main__':
    for mode in ['test']:
        main(mode)