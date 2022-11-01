import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
import os
import re

path = '/opt/ml/input/code/data/'
books = pd.read_csv(path + 'books.csv')
users = pd.read_csv(path + 'users.csv')
ratings = pd.read_csv(path + 'train_ratings.csv')
test_ratings = pd.read_csv(path + 'test_ratings.csv')


def main():
    # language (버리는 행이라 -1) => 경우의 수 : 1
    books.rename(columns={'language' : 'language1'}, inplace=True)
    books['language1'] = -1 
    #book_author => 경우의 수 : 2
    books.rename(columns={'book_author' : 'book_author1'}, inplace=True)
    books['book_author2'] = -1 # author 안쓰는 column

    # summary => 경우의 수 : 1
    books.rename(columns={'summary' : 'summary1'}, inplace=True)
    books['summary1'] = books['summary1'].fillna('None') #summary 결측치 None로

    # category => 경우의 수 : 3
    books.rename(columns={'category' : 'category1'}, inplace=True)
    books.loc[books[books['category1'].notnull()].index, 'category1'] = books[books['category1'].notnull()]['category1'].apply(lambda x: re.sub('[\W_]+',' ',x).strip())
    books['category1'] = books['category1'].str.lower()

    books['category2'] = books['category1']
    books['category3'] = books['category1']

    # 미션 category EDA 그대로

    books['category_high1'] = books['category1'].copy()
    categories1 = ['garden','crafts','physics','adventure','music','fiction','nonfiction','science','science fiction','social','homicide',
'sociology','disease','religion','christian','philosophy','psycholog','mathemat','agricult','environmental',
'business','poetry','drama','literary','travel','motion picture','children','cook','literature','electronic',
'humor','animal','bird','photograph','computer','house','ecology','family','architect','camp','criminal','language','india']

    for category in categories1:
        books.loc[books[books['category1'].str.contains(category,na=False)].index,'category_high1'] = category
    
    category_high_df1 = pd.DataFrame(books['category_high1'].value_counts()).reset_index()
    category_high_df1.columns = ['category1','count']
    # 5개 이하인 항목은 others로 묶어주도록 하겠습니다.
    others_list = category_high_df1[category_high_df1['count']<5]['category1'].values
    books.loc[books[books['category_high1'].isin(others_list)].index, 'category_high1']='others'

    
    # bookcrossing 사이트 카테고리 사용
    books['category_high2'] = books['category2'].copy()
    categories2 = ['garden','crafts','physics','adventure','music','fiction','nonfiction','science','science fiction','social','homicide',
'sociology','disease','religion','christian','philosophy','psycholog','mathemat','agricult','environmental',
'business','poetry','drama','literary','travel','motion picture','children','cook','literature','electronic',
'humor','animal','bird','photograph','computer','house','ecology','family','architect','camp','criminal','language','india', 'sports', 'horror', 'health']

    for category in categories2:
        books.loc[books[books['category2'].str.contains(category,na=False)].index,'category_high2'] = category
    
    category_high_df2 = pd.DataFrame(books['category_high1'].value_counts()).reset_index()
    category_high_df2.columns = ['category2','count']
    # 5개 이하인 항목은 others로 묶어주도록 하겠습니다.
    others_list = category_high_df2[category_high_df2['count']<5]['category2'].values
    books.loc[books[books['category_high2'].isin(others_list)].index, 'category_high2']='others'
    
    # category : fiction & nonfiction
    books.loc[books['category3'] == 'juvenile fiction','category3'] = 'fiction'
    books.loc[books['category3'] != 'fiction', 'category3'] = 'nonfiction'


    books['category1'] = books['category_high1']
    books['category2'] = books['category_high2']
    books['category1'] = books['category1'].fillna('None')
    books['category2'] = books['category2'].fillna('None')
    books['category3'] = books['category3'].fillna('None')
    books.drop('category_high1', axis = 1, inplace=True)
    books.drop('category_high2', axis = 1, inplace=True)

    


    # publisher => 경우의 수 : 2개
    books.rename(columns={'publisher' : 'publisher1'}, inplace=True)

    # # publisher 있는경우
    publisher_dict=(books['publisher1'].value_counts()).to_dict()
    publisher_count_df= pd.DataFrame(list(publisher_dict.items()),columns = ['publisher1','count'])
    publisher_count_df = publisher_count_df.sort_values(by=['count'], ascending = False)
    modify_list = publisher_count_df[publisher_count_df['count']>5].publisher1.values
    for publisher1 in modify_list:
        try:
            number = books[books['publisher1']==publisher1]['isbn'].apply(lambda x: x[:4]).value_counts().index[0]
            right_publisher = books[books['isbn'].apply(lambda x: x[:4])==number]['publisher1'].value_counts().index[0]
            books.loc[books[books['isbn'].apply(lambda x: x[:4])==number].index,'publisher1'] = right_publisher
        except:
            pass

    books['publisher2'] = -1  # publisher 없는경우

    # year_of_publication => 경우의 수 : 2개
    books.rename(columns={'year_of_publication' : 'year_of_publication1'}, inplace=True) #year 있는경우
    books['year_of_publication2'] = -1 # year 없는경우


    if not os.path.exists('../../data'):
        os.makedirs('../../data')
    if not os.path.exists('../../data/books'):
        os.makedirs('../../data/books')


    n_year = 2
    n_pub = 2
    n_cat = 3
    n_auth = 2


    num = 1
    for i in range(n_year):
        for j in range(n_pub):
            for k in range(n_cat):
                for l in range(n_auth):
                    formatted_num = format(num, '02')
                    file = books[['isbn', 'book_title', f'book_author{l+1}', f'year_of_publication{i+1}',
                     f'publisher{j+1}', 'img_url', 'language1', f'category{k+1}', 'summary1', 'img_path']]
                    file.columns = [['isbn', 'book_title', 'book_author', 'year_of_publication',
                     'publisher', 'img_url', 'language', 'category', 'summary', 'img_path']]
                    file.to_csv(f'../../data/books/b{formatted_num}.csv', index=False)
                    num += 1


if __name__ == '__main__':
    main()