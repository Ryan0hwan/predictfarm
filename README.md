# Predict agriculture price
5대 농산물의 데이터를 수집하여, 인공신경망 모델(RNN, LSTM, GRU) 중 각각의 데이터에 가장 좋은 성능을 보이는 모델을 선택하고, 
그 모델 안에서, 어떤 변수가 가장 가격예측에 큰 영향을 끼치는지 shap라이브러리를 통해 확인하고,  
해당 모델을 통해 실제 가격을 예측하는 프로젝트입니다.  
5대 농산물 : 배추, 양파, 마늘, 무, 고추 (한국농촌경제원에 지정)  
[농산물가격예측.pdf](https://github.com/user-attachments/files/18297375/default.pdf)
  
# 1. 데이터 수집
* 출처 : KAMIS, 기상자료개방포털, INVESTING, 농넷, KOSIS
* 14년 1월 1주차 ~ 24년 8월 5주차
* 12개의 독립변수로 1개의 종속변수(소매가)예측  
![데이터소개](https://github.com/user-attachments/assets/24f35549-4542-4012-a5fc-ec59984a46a4)  
  
# 2. 데이터 전처리
* 금액/무게 단위 통합 (원, 달러 -> 원 / t, g -> kg)
* 주간 데이터로 변환 (일간,주간,월간,연간 -> 주간)
* 결측치 처리 : 결측치가 많은 열은 열 삭제, 결측치가 적은 열은 열 전체의 평균값으로 대체
* 정규화(MinMaxScaler)
* Feature값 선정(f_regression, SFS, rfe, rfecv 이렇게 4가지 방법을 사용하여 나온 변수들 중, 최소 2가지 방법에서 나온 변수를 feature로 선정)  
![image](https://github.com/user-attachments/assets/f5dc06cc-12bf-4451-9316-63f53aa8b559)  
  
# 3. 모델링
* 파라미터와 함수 설정([LSTM 네트워크를 활용한 농산물 가격 예측 모델(2018.11)](https://scienceon.kisti.re.kr/commons/util/originalView.do?cn=JAKO201809469053682&oCn=JAKO201809469053682&dbt=JAKO&journal=NJOU00292001) 참고)  
![모델링 요약](https://github.com/user-attachments/assets/322dc70d-3c3c-47d0-a2d1-560da29b3dfd)  
![image](https://github.com/user-attachments/assets/aff2cf6b-f3f2-41b3-a059-97a90838b91e)  
  
# 4. 5대 농산물별 RNN, LSTM, GRU 모델링 성능 결과 비교 (그래프/ MAE, MAPE, RMSE)
![image](https://github.com/user-attachments/assets/55c4d2e2-0849-4a68-a1f8-a408e9608a9e)
![image](https://github.com/user-attachments/assets/0d0eeb9a-0074-4fab-a129-41e870650df5)
![image](https://github.com/user-attachments/assets/7b00fad1-74b1-467c-9c9c-92c4d482f1f2)
![image](https://github.com/user-attachments/assets/c93d50cb-5e60-489d-a9d8-917cfd4997aa)
![image](https://github.com/user-attachments/assets/d0048291-4ebb-48a4-9993-4d45bc5f27fb)
![image](https://github.com/user-attachments/assets/189b35c5-bd63-4120-82b8-58ca7d887907)  
* 배추, 마늘: GRU를 가격예측모델로 선정
* 양파, 무, 고추: LSTM을 가격예측모델로 선정  
  
# 5. 5대 농산물별 가격 예측에 가장 큰 영향을 끼치는 변수 파악 (SHAP)
### 배추
![image](https://github.com/user-attachments/assets/f979f165-e4e1-47f7-b511-a865c2d32e0f)
### 양파
![image](https://github.com/user-attachments/assets/4017acaf-f458-4965-9f73-9fb3cbb3ca6d)
### 마늘
![image](https://github.com/user-attachments/assets/7b1a0b95-ea32-4005-a10d-75dcbb9790ae)
### 무
![image](https://github.com/user-attachments/assets/7c983f98-4bfd-4bdc-b13d-142bff19c7c4)
### 고추
![image](https://github.com/user-attachments/assets/3ef2b1ba-e6d3-4c96-91e4-032d7265f66f)

# 6. 가격 예측
![image](https://github.com/user-attachments/assets/e60edf69-991a-42ab-b847-2ab6dad9fde1)
![image](https://github.com/user-attachments/assets/a89258cb-36c1-46cf-af1b-4956ebb69e99)  
  
# 7. 한계점
1. 농산물의가격예측그래프를보면, 평균근처의값들은잘예측하나, 특정값이상의outlier 값들의경우잘예측하지못한다.
2. 시간이 장기화 될수록, 모델의 예측 성능이 떨어졌다.  
 (1달 뒤 가격 예측 결과, 좋은 성능을 보였고, 2달 뒤 가격 예측도 준수한 가격 예측을 보였으나, 3달 뒤부터 가격 예측 성능이 많이 빗나갔다.)
