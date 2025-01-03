# Predict agriculture price
5대 농산물의 데이터를 수집하여, 인공신경망 모델(RNN, LSTM, GRU) 중 각각의 데이터에 가장 좋은 성능을 보이는 모델을 선택하고, 
해당 모델을 통해 실제 가격을 예측하는 프로젝트입니다.
5대 농산물 : 배추, 양파, 마늘, 무, 고추 (한국농촌경제원에 지정)

[농산물가격예측.pdf](https://github.com/user-attachments/files/18297375/default.pdf)

## 1. 데이터 수집
* 출처 : KAMIS, 기상자료개방포털, INVESTING, 농넷, KOSIS
* 14년 1월 1주차 ~ 24년 8월 5주차
* 12개의 독립변수로 1개의 종속변수(소매가)예측
![데이터소개](https://github.com/user-attachments/assets/24f35549-4542-4012-a5fc-ec59984a46a4)  

## 2. 데이터 전처리
* 금액/무게 단위 통합 (원, 달러 -> 원 / t, g -> kg)
* 주간 데이터로 변환 (일간,주간,월간,연간 -> 주간)
* 결측치 처리 : 결측치가 많은 열은 열 삭제, 결측치가 적은 열은 열 전체의 평균값으로 대체
* 정규화(MinMaxScaler)
* Feature값 선정(f_regression, SFS, rfe, rfecv 이렇게 4가지 방법을 사용하여 나온 변수들 중, 최소 2가지 방법에서 나온 변수를 feature로 선정)  




  
