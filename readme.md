<img width="1020" alt="AITech배너" src="https://user-images.githubusercontent.com/107118671/200157171-0a8d95d6-9994-4fd1-8a07-8645a796d3cd.png">

# 책 평점 예측 (Book Rating Prediction)

## 주제 설명
일반적으로 책 한 권은 원고지 기준 800~1000매 정도 되는 분량을 가지고 있습니다.

뉴스기사나 짧은 러닝 타임의 동영상처럼 간결하게 즐길 수 있는 '숏폼 컨텐츠'는 소비자들이 부담 없이 쉽게 선택할 수 있지만, 
책 한 권을 모두 읽기 위해서는 보다 긴 물리적인 시간이 필요합니다.

또한 소비자 입장에서는 제목, 저자, 표지, 카테고리 등 한정된 정보로 각자가 콘텐츠를 유추하고 구매 유무를 결정해야 하기 때문에 상대적으로 더욱 신중을 가하게 됩니다.

본 프로젝트는 이러한 소비자들의 책 구매 결정에 대한 도움을 주기 위한 개인화된 상품 추천 모델을 설계하였습니다.

책과 관련된 정보와 소비자의 정보, 그리고 소비자가 실제로 부여한 평점, 총 세 가지의 데이터셋(`users.csv`, `books.csv`, `train_ratings.csv`)을 
활용하여 각 사용자가 주어진 책에 대해 어떤 평점을 부여할 지에 대해 예측해 보았습니다.

## Environment
```
catboost==1.1
implicit==0.6.1
lightfm==1.16
lightgbm==3.3.3
matplotlib==3.6.0
matplotlib-inline==0.1.6
plotly==5.10.0
PyYAML==5.4.1
recommenders==1.1.1
scikit-surprise==1.1.3
seaborn==0.12.1
surprise==0.1
xgboost==1.6.2
```
## Directory

📦level1_bookratingprediction_recsys-level1-recsys-11

 ┣ 📂experiments
 
 ┣ 📂preprocess
 
 ┣ 📂src
 
 ┣ 📜ensemble.py
 
 ┗ 📜main.py

## Dataset

학습 데이터는 306,795건의 평점 데이터(`train_rating.csv`)이며, 
149,570건의 책 정보(`books.csv`) 및 
68,092명의 고객 정보(`users.csv`)를 이용했습니다.

각각 데이터는 다음의 형태를 띄고 있습니다.

- input

  - `train_ratings.csv` : 각 사용자가 책에 대해 평점을 매긴 내역

  - `users.csv` : 사용자에 대한 정보

  - `books.csv` : 책에 대한 정보

  - `Image/` : 책 이미지
  
> train_ratings.csv
![image](https://user-images.githubusercontent.com/107118671/200157486-6492725c-9719-4d20-af1c-06ac03634013.png)

> users.csv
![image](https://user-images.githubusercontent.com/107118671/200157492-280b0a07-9d46-4c3a-8a98-065a5b2d8c8d.png)

> books.csv
![image](https://user-images.githubusercontent.com/107118671/200157506-cbaacbf3-3070-4efd-a07a-bf9bb3e1a4a5.png)

> image data
![image](https://user-images.githubusercontent.com/107118671/200157509-72a04c0c-1875-40c0-bc94-cb05c41e5d55.png)

평가 데이터는 총 76,699건의 평점 데이터로 구성돼 있습니다.

- output
  - `test_ratings.csv`의 사용자가 주어진 책에 대해 매길 것이라고 예상하는 평점

## EDA

## Model

## References
