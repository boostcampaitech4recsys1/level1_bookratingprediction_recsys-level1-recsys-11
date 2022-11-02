import os
import re
import nltk
from nltk import tokenize
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from transformers import BertModel, BertTokenizer


# def text_preprocessing(summary):
#     summary = re.sub("[.,\'\"''""!?]", "", summary)
#     summary = re.sub("[^0-9a-zA-Z\\s]", " ", summary)
#     summary = re.sub("\s+", " ", summary)
#     summary = summary.lower()
#     return summary


# def summary_merge(df, user_id, max_summary):
#     return " ".join(df[df['user_id'] == user_id].sort_values(by='summary_length', ascending=False)['summary'].values[:max_summary])


def age_map(x: int) -> int:
    x = int(x)
    if x < 20:
        return 1
    elif x >= 20 and x < 30:
        return 2
    elif x >= 30 and x < 40:
        return 3
    elif x >= 40 and x < 50:
        return 4
    elif x >= 50 and x < 60:
        return 5
    else:
        return 6


def year_of_publication_map(x: int) -> int:
    x = int(x)
    if x < 1991:
        return 1
    elif x >= 1991 and x < 1996:
        return 2
    elif x >= 1996 and x < 2000:
        return 3
    else:
        return 4


# def text_to_vector(text, tokenizer, model, device):
#     for sent in tokenize.sent_tokenize(text):
#         text_ = "[CLS] " + sent + " [SEP]"
#         tokenized = tokenizer.tokenize(text_)
#         indexed = tokenizer.convert_tokens_to_ids(tokenized)
#         segments_idx = [1] * len(tokenized)
#         token_tensor = torch.tensor([indexed])
#         sgments_tensor = torch.tensor([segments_idx])
#         with torch.no_grad():
#             outputs = model(token_tensor.to(device), sgments_tensor.to(device))
#             encode_layers = outputs[0]
#             sentence_embedding = torch.mean(encode_layers[0], dim=0)
#     return sentence_embedding.cpu().detach().numpy()

def process_context_data(users, books, ratings1, ratings2):
    ratings = pd.concat([ratings1, ratings2]).reset_index(drop=True)

    # 인덱싱 처리된 데이터 조인
    context_df = ratings.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'year_of_publication', 'book_author']], on='isbn', how='left')
    train_df = ratings1.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'year_of_publication', 'book_author']], on='isbn', how='left')
    test_df = ratings2.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'year_of_publication', 'book_author']], on='isbn', how='left')

    column_list = ['location_city', 'location_state', 'location_country', 'age', 'book_author', \
                        'year_of_publication', 'publisher', 'category']

    columns = [column for column in column_list if train_df.iloc[0][column] != -1]

    # 인덱싱 처리
    loc_city2idx = {v:k for k,v in enumerate(context_df['location_city'].unique())}
    loc_state2idx = {v:k for k,v in enumerate(context_df['location_state'].unique())}
    loc_country2idx = {v:k for k,v in enumerate(context_df['location_country'].unique())}

    train_df['location_city'] = train_df['location_city'].map(loc_city2idx)
    train_df['location_state'] = train_df['location_state'].map(loc_state2idx)
    train_df['location_country'] = train_df['location_country'].map(loc_country2idx)
    test_df['location_city'] = test_df['location_city'].map(loc_city2idx)
    test_df['location_state'] = test_df['location_state'].map(loc_state2idx)
    test_df['location_country'] = test_df['location_country'].map(loc_country2idx)

    train_df['age'] = train_df['age'].apply(age_map)
    test_df['age'] = test_df['age'].apply(age_map)

    # year of publication cases 4
    train_df['year_of_publication'] = train_df['year_of_publication'].apply(year_of_publication_map)
    test_df['year_of_publication'] = test_df['year_of_publication'].apply(year_of_publication_map)

    # book 파트 인덱싱
    category2idx = {v:k for k,v in enumerate(context_df['category'].unique())}
    publisher2idx = {v:k for k,v in enumerate(context_df['publisher'].unique())}
    author2idx = {v:k for k,v in enumerate(context_df['book_author'].unique())}

    train_df['category'] = train_df['category'].map(category2idx)
    train_df['publisher'] = train_df['publisher'].map(publisher2idx)
    train_df['book_author'] = train_df['book_author'].map(author2idx)
    test_df['category'] = test_df['category'].map(category2idx)
    test_df['publisher'] = test_df['publisher'].map(publisher2idx)
    test_df['book_author'] = test_df['book_author'].map(author2idx)

    idx = {
        "loc_city2idx":loc_city2idx,
        "loc_state2idx":loc_state2idx,
        "loc_country2idx":loc_country2idx,
        "category2idx":category2idx,
        "publisher2idx":publisher2idx,
        "author2idx":author2idx,
    }

    length_db = {'age':6, 'year_of_publication': 4, 
                'location_city': len(idx['loc_city2idx']), 'location_state': len(idx['loc_state2idx']), 
                'location_country': len(idx['loc_country2idx']), 'category': len(idx['category2idx']), 'publisher': len(idx['publisher2idx']),
                'book_author': len(idx['author2idx'])}
    field_dims = [length_db[column] for column in columns]

    return idx, train_df, test_df, columns, field_dims


def process_text_data(df, user2idx, isbn2idx, device, train=False):
    # df_ = df.copy()
    print('Check Vectorizer')
    print('Vector Load')
    print("[USER]")
    if train == True:
        user = np.load('/opt/ml/data/embedding/train/user_summary_merge_vector.npy', allow_pickle = True)
    else:
        user = np.load('/opt/ml/data/embedding/test/user_summary_merge_vector.npy', allow_pickle = True)
    user_review_text_df = pd.DataFrame([user[0], user[1]]).T
    user_review_text_df.columns = ['user_id', 'user_summary_merge_vector']
    user_review_text_df['user_id'] = user_review_text_df['user_id'].map(user2idx)
    df = pd.merge(df, user_review_text_df, on='user_id', how='left')
    del user
    del user_review_text_df

    print("[ITEM]")
    if train == True:
        item = np.load('/opt/ml/data/embedding/train/item_summary_vector.npy', allow_pickle = True)
    else:
        item = np.load('/opt/ml/data/embedding/test/item_summary_vector.npy', allow_pickle = True)
    books_text_df = pd.DataFrame([item[0], item[1]]).T
    books_text_df.columns = ['isbn', 'item_summary_vector']
    books_text_df['isbn'] = books_text_df['isbn'].map(isbn2idx)
    del item
    df = pd.merge(df, books_text_df[['isbn', 'item_summary_vector']], on='isbn', how='left')
    del books_text_df

    return df


class Text_Dataset(Dataset):
    def __init__(self, user_isbn_vector, user_summary_merge_vector, item_summary_vector, label):
        self.user_isbn_vector = user_isbn_vector
        self.user_summary_merge_vector = user_summary_merge_vector
        self.item_summary_vector = item_summary_vector
        self.label = label

    def __len__(self):
        return self.user_isbn_vector.shape[0]

    def __getitem__(self, i):
        return {
                'user_isbn_vector' : torch.tensor(self.user_isbn_vector[i], dtype=torch.long),
                'user_summary_merge_vector' : torch.tensor(self.user_summary_merge_vector[i].reshape(-1, 1), dtype=torch.float32),
                'item_summary_vector' : torch.tensor(self.item_summary_vector[i].reshape(-1, 1), dtype=torch.float32),
                'label' : torch.tensor(self.label[i], dtype=torch.float32),
                }


def text_data_load(args):

    formatted_user_num = format(args.USER_NUM, '02')
    formatted_book_num = format(args.BOOK_NUM, '02')
    users = pd.read_csv(args.DATA_PATH + 'users/' + f'u{formatted_user_num}.csv')
    books = pd.read_csv(args.DATA_PATH + 'books/' + f'b{formatted_book_num}.csv')
    train = pd.read_csv(args.DATA_PATH + 'ratings/' + 'train_ratings.csv')
    test = pd.read_csv(args.DATA_PATH + 'ratings/' + 'test_ratings.csv')
    sub = pd.read_csv(args.DATA_PATH + 'ratings/' + 'sample_submission.csv')

    ids = pd.concat([train['user_id'], sub['user_id']]).unique()
    isbns = pd.concat([train['isbn'], sub['isbn']]).unique()

    idx2user = {idx:id for idx, id in enumerate(ids)}
    idx2isbn = {idx:isbn for idx, isbn in enumerate(isbns)}

    user2idx = {id:idx for idx, id in idx2user.items()}
    isbn2idx = {isbn:idx for idx, isbn in idx2isbn.items()}

    train['user_id'] = train['user_id'].map(user2idx)
    sub['user_id'] = sub['user_id'].map(user2idx)
    test['user_id'] = test['user_id'].map(user2idx)
    users['user_id'] = users['user_id'].map(user2idx)

    train['isbn'] = train['isbn'].map(isbn2idx)
    sub['isbn'] = sub['isbn'].map(isbn2idx)
    test['isbn'] = test['isbn'].map(isbn2idx)
    books['isbn'] = books['isbn'].map(isbn2idx)

    idx, context_train, context_test, columns, field_dims = process_context_data(users, books, train, test)


    print("[TEXT TRAIN]")
    text_train = process_text_data(context_train, user2idx, isbn2idx, args.DEVICE, train=True)
    print("[TEXT TEST]")
    text_test = process_text_data(context_test, user2idx, isbn2idx, args.DEVICE, train=False)

    columns = ['user_id', 'isbn'] + columns
    field_dims = np.array([len(user2idx), len(isbn2idx)] + field_dims, dtype = np.int64)
    text_train = text_train[columns + ['user_summary_merge_vector', 'item_summary_vector'] + ['rating']]
    text_test = text_test[columns + ['user_summary_merge_vector', 'item_summary_vector'] + ['rating']]

    print(text_train.info(), '\n\n')
    print(text_test.info())
    data = {
            'train':train,
            'test':test,
            'users':users,
            'books':books,
            'sub':sub,
            'idx2user':idx2user,
            'idx2isbn':idx2isbn,
            'user2idx':user2idx,
            'isbn2idx':isbn2idx,
            'text_train':text_train,
            'text_test':text_test,
            'field_dims': field_dims,
            'columns': columns
            }

    return data


def text_data_split(args, data):
    X_train, X_valid, y_train, y_valid = train_test_split(
                                                        data['text_train'].drop(['rating'], axis=1),
                                                        data['text_train']['rating'],
                                                        test_size=args.TEST_SIZE,
                                                        random_state=args.SEED,
                                                        shuffle=True
                                                        )
    data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
    print("[SAMPLING]")
    print(data['X_train'].columns)
    print(data['X_train'].sample(5))
    return data


def text_data_loader(args, data):
    train_dataset = Text_Dataset(
                                data['X_train'][data['columns']].values,
                                data['X_train']['user_summary_merge_vector'].values,
                                data['X_train']['item_summary_vector'].values,
                                data['y_train'].values
                                )
    valid_dataset = Text_Dataset(
                                data['X_valid'][data['columns']].values,
                                data['X_valid']['user_summary_merge_vector'].values,
                                data['X_valid']['item_summary_vector'].values,
                                data['y_valid'].values
                                )
    test_dataset = Text_Dataset(
                                data['text_test'][data['columns']].values,
                                data['text_test']['user_summary_merge_vector'].values,
                                data['text_test']['item_summary_vector'].values,
                                data['text_test']['rating'].values
                                )


    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=True, num_workers = 4)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.BATCH_SIZE, shuffle=True, num_workers = 4)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.BATCH_SIZE, shuffle=False, num_workers = 4)
    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader

    return data
