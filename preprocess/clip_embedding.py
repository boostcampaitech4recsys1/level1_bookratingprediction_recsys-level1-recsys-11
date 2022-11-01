import os
import clip
from PIL import Image
import torch
from tqdm import tqdm
import pandas as pd
import re
import numpy as np
import pickle

NONE_TENSOR = np.load("/opt/ml/data/none_tensor.npy")


def embed_image(book_df, model, preprocess, device):
    img_vector_df = book_df[['img_path']].drop_duplicates().reset_index(drop = True).copy()
    data_box = []
    for name in tqdm(img_vector_df['img_path']):
        image_path = os.path.join("/opt/ml/input/code/data", name)
        image = Image.open(image_path).convert('RGB')
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features = image_features.squeeze().detach().cpu().numpy()
        data_box.append(image_features)
    with open("/opt/ml/data/image_feature_vector.pkl", 'wb') as f:
        pickle.dump(data_box, f)
    img_vector_df['image_embed'] = data_box
    book_df = pd.merge(book_df, img_vector_df, on = 'img_path', how = 'left')
    return book_df


def text_preprocessing(summary):
    summary = re.sub("[.,\'\"''""!?]", "", summary)
    summary = re.sub("[^0-9a-zA-Z\\s]", " ", summary)
    summary = re.sub("\s+", " ", summary)
    summary = summary.lower()
    return summary


def summary_merge(df, user_id, max_summary):
    temp = " ".join(df[df['user_id'] == user_id].sort_values(by='summary_length', ascending=False)['summary'].values[:max_summary])
    return temp


def text_to_vector(text, model, device):
    try:
        text = clip.tokenize(text).to(device)
        # print(text.shape)
        with torch.no_grad():
            text_features = model.encode_text(text)
            # print(text_features.shape)
        return text_features.squeeze().detach().cpu().numpy()
    
    except:
        return NONE_TENSOR
    
    
            
def main():
    root_dir = "/opt/ml/data/books"
    device = "cuda:0"
    print("[LOAD CLIP]")
    model, preprocess = clip.load("ViT-B/32", device = device)

    book_df = pd.read_csv(os.path.join(root_dir, 'b01.csv'))
    rating_df = pd.read_csv(os.path.join("/opt/ml/data/ratings", 'train_ratings.csv'))
    book_df['summary'] = book_df['summary'].apply(lambda x:text_preprocessing(x))
    book_df['book_title'] = book_df['book_title'].apply(lambda x: text_preprocessing(x))
    df_fe = pd.merge(rating_df, book_df[['isbn', 'book_title', 'summary']], how = 'inner', on = 'isbn')

    df_fe['book_title_length'] = df_fe['book_title'].apply(lambda x: len(x))
    df_fe['summary_length'] = df_fe['summary'].apply(lambda x: len(x))
    print(df_fe.sample(5).columns)

    print(f"[USER SUMMARY MERGE VECTOR]")
    ppath = "/opt/ml/data/user_summary_merge_vector.pkl"
    if os.path.exists(ppath):
        with open(ppath, 'rb') as f:
            user_summary_merge_vector = pickle.load(f)
        # user_review_text_df = pd.DataFrame(df_fe['user_id'].unique(), columns=['user_id'])
    else:
        user_summary_merge_vector = []
        for x in tqdm(df_fe['user_id'].unique()):
            user_summary_merge_vector.append(text_to_vector(summary_merge(df_fe, x, 1), model, device))
        # user_review_text_df = pd.DataFrame(df_fe['user_id'].unique(), columns=['user_id'])

        with open("ppath", 'wb') as f:
            pickle.dump(user_summary_merge_vector, f)
    user_review_text_df = pd.DataFrame(df_fe['user_id'].unique(), columns=['user_id'])
    # print(type(user_review_text_df), user_review_text_df.columns)

    print(f"[BOOK TITLE VECTOR]")
    ppath = "/opt/ml/data/book_title_vector.pkl"
    if os.path.exists(ppath):
        with open(ppath, 'rb') as f:
            book_title_vector = pickle.load(f)
    else:
        book_title_vector = []
        for x in tqdm(book_df['book_title'].tolist()):
            book_title_vector.append(text_to_vector(x, model, device))
        with open("/opt/ml/data/book_title_vector.pkl", 'wb') as f:
            pickle.dump(book_title_vector, f)


    print("[BOOK SUMMARY VECTOR]")
    ppath = "/opt/ml/data/item_summary_vector.pkl"
    if os.path.exists(ppath):
        with open(ppath, 'rb') as f:
            item_summary_vector = pickle.load(f)
    else:
        item_summary_vector = []
        for x in tqdm(book_df['summary'].tolist()):
            item_summary_vector.append(text_to_vector(x, model, device))
        with open("/opt/ml/data/item_summary_vector.pkl", 'wb') as f:
            pickle.dump(item_summary_vector, f)

    # print(user_review_text_df.shape, user_review_text_df.columns)
    user_review_text_df['user_summary_merge_vector'] = user_summary_merge_vector
    book_df['book_title_vector'] = book_title_vector
    book_df['item_summary_vector'] = item_summary_vector


    df_fe_join = pd.merge(df_fe, user_review_text_df, on='user_id', how='left')
    df_fe_join = pd.merge(df_fe_join, book_df[['isbn', 'img_path', 'book_title_vector', 'item_summary_vector']], on='isbn', how='left')
    print(df_fe_join.sample(5).columns)

    print("[BOOK IMAGE VECTOR]")
    book_df = embed_image(df_fe_join, model, preprocess, device)
    book_df.to_csv(f'/opt/ml/data/embedding.csv', index=False)


if __name__ == '__main__':
    main()