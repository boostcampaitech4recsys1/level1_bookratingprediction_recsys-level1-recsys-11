import pandas as pd
import numpy as np
from tqdm import tqdm

import os
import pickle

def age1(users: pd.DataFrame, ratings, test_ratings):
    no_age_users = users[users['age'].isna()]['user_id'].unique()
    rating_users = pd.concat([ratings, test_ratings])['user_id'].unique()

    no_age_yes_rating = list(set(no_age_users) & set(rating_users))

    def get_age(
        book_list: list, # 북 리스트는 어떤 유저가 읽은 책 리스트
        ratings: pd.DataFrame,
        users: pd.DataFrame,
        isbn_age: dict
    ):

        expected_ages = []
        
        for isbn in book_list:
            if isbn in isbn_age:
                expected_ages.append(isbn_age[isbn])
                continue
            yes_read = ratings[ratings['isbn'] == isbn]['user_id'].unique() 
            yes_age = users[users['age'].notnull()] 
            expected_age = yes_age[yes_age['user_id'].isin(yes_read)]['age'].median() 
            isbn_age[isbn] = expected_age 
            expected_ages.append(expected_age)

        expected_ages = [e for e in expected_ages if not np.isnan(e)] 

        if not expected_ages: 
            return np.nan
        return np.round(np.median(expected_ages))

    isbn_age = {}
    try:
        f = open('isbn_age.pkl', 'rb')
        isbn_age = pickle.load(f)
        f.close()
    except:
        pass
    
    user_age = {}
    for user_id in tqdm(no_age_yes_rating):
        books = ratings[ratings['user_id']==user_id]['isbn'].values
        user_age[user_id] = get_age(books, ratings, users, isbn_age)

    f = open('isbn_age.pkl', 'wb')
    pickle.dump(isbn_age, f)
    f.close()

    # age 복사 후 새 컬럼에 복제하고, 결측채움
    users['age1'] = users['age'].copy()
    for uid, age in user_age.items():
        users.loc[users['user_id'] == uid, 'age1'] = age

    # 나머지는 age의 평균값으로 채움
    users.loc[users['age1'].isna(), 'age1'] = np.round(users['age'].mean())
    print('age1 done')
    return users


def age2(users: pd.DataFrame) -> pd.DataFrame:
    # 올 평균
    users['age2'] = users['age'].copy()
    users.loc[users['age2'].isna(), 'age2'] = np.round(users['age'].mean())
    print('age2 done')
    return users


def age3(users: pd.DataFrame) -> pd.DataFrame:
    # 국가별 평균
    users['age3'] = users['age'].copy()
    countries = users['location_country'].value_counts().index
    for country in countries:
        mean_age = users[users['location_country'] == country]['age'].mean()
        users.loc[(users['age'].isna()) & (users['location_country'] == country), 'age3'] = np.round(mean_age)

    users.loc[users['age3'].isna(), 'age3'] = np.round(users['age'].mean())
    print('age3 done')
    return users


def loc1(users: pd.DataFrame) -> pd.DataFrame:
    # 기본 미션 eda
    users['location_city1'] = users['location_city'].copy()
    users['location_state1'] = users['location_state'].copy()
    users['location_country1'] = users['location_country'].copy()
    print('loc1 done')
    return users


def loc2(users: pd.DataFrame) -> pd.DataFrame:
    # usa, others, None
    users['location_city2'] = -1
    users['location_state2'] = -1
    users['location_country2'] = users['location_country'].copy()

    users.loc[(users['location_country'] != 'None') & (users['location_country'] != 'usa'), 'location_country2'] = 'others'
    print('loc2 done')
    return users


def loc3(users: pd.DataFrame) -> pd.DataFrame:
    # 국가들 모두 살림, None
    users['location_city3'] = -1
    users['location_state3'] = -1
    users['location_country3'] = users['location_country'].copy()

    print('loc3 done')
    return users

def main():
    # 전체 파일 준비
    users = pd.read_csv('../../input/code/data/users.csv')
    books = pd.read_csv('../../input/code/data/books.csv')
    ratings = pd.read_csv('../../input/code/data/train_ratings.csv')
    test_ratings = pd.read_csv('../../input/code/data/test_ratings.csv')

    # users location을 city, state, country로 나누는 기본 작업
    users['location'] = users['location'].str.replace(r'[^0-9a-zA-Z:,]', '') # 특수문자 제거
    users['location_city'] = users['location'].apply(lambda x: x.split(',')[0].strip())
    users['location_state'] = users['location'].apply(lambda x: x.split(',')[1].strip())
    users['location_country'] = users['location'].apply(lambda x: x.split(',')[2].strip())

    users = users.replace('na', 'None')
    users = users.replace('', 'None')

    modify_location = users[(users['location_country'].isna())&(users['location_city'].notnull())]['location_city'].values

    location_list = []
    for location in modify_location:
        try:
            right_location = users[(users['location'].str.contains(location))&(users['location_country'].notnull())]['location'].value_counts().index[0]
            location_list.append(right_location)
        except:
            pass
    
    for location in location_list:
        users.loc[users[users['location_city']==location.split(',')[0]].index,'location_state'] = location.split(',')[1]
        users.loc[users[users['location_city']==location.split(',')[0]].index,'location_country'] = location.split(',')[2]

    # 경우의 수 시작
    users = loc1(users) # eda 처리만, 그대로 복사해서 행 추가
    users = loc2(users) # usa는 usa, 다른나라는 '-' 로 진행, 시티, 스테잍은 버린다. -> 시티, 스테잍은 -1
    users = loc3(users) # 국가들 모두 살리고 시티 스테잍은 -1

    users = age1(users, ratings, test_ratings) # age1 컬럼 추가
    users = age2(users)
    users = age3(users)

    n_loc = 3
    n_age = 3

    if not os.path.exists('../../data'):
        os.makedirs('../../data')
    if not os.path.exists('../../data/users'):
        os.makedirs('../../data/users')


    num = 1
    for i in range(n_loc):
        for j in range(n_age):
            file = users[['user_id', f'location_city{i+1}', f'location_state{i+1}', f'location_country{i+1}', f'age{j+1}']]
            file.columns = [['user_id', 'location_city', 'location_state', 'location_country', 'age']]
            file.to_csv(f'../../data/users/u{num}.csv', index=False)
            num += 1


if __name__ == '__main__':
    main()