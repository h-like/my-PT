# 파이썬 패키지 수입
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time

import torch
from  torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

# 하이퍼 파라미터
INPUT_DIM = 13
MY_HIDDEN = 1000
MY_EPOCH = 1000

# 추가 옵션
pd.set_option('display.max_columns', None)
torch.manual_seed(111)
import numpy as np
np.random.seed(111)

# 데이터 파일 읽기
# 결과는 pandas의 데이터 프레임 형식
raw = pd.read_csv('heart.csv')

print('원본 데이터 샘플 10개')
print(raw.head(10))
print('원본 데이터 통계')
print(raw.describe())

# 데이터를 입력과 출력으로 분리
X_data = raw.drop('target', axis=1)
Y_data = raw['target']
names = X_data.columns
print(names)

# 데이터를 학습용과 평가용으로 분리
X_train, X_test, Y_train, Y_test = \
    train_test_split(X_data, Y_data,
                     test_size=0.3)

# 최종 데이터 모양
print('\n학습용 입력 데이터 모양:', X_train.shape)
print('학습용 출력 데이터 모양', Y_train.shape)
print('평가용 입력 데이터 모양:', X_test.shape)
print('평가용 출력 데이터 모양:', Y_test.shape)

# 입력 데이터 Z-점수 정규화
# 결과는 numpy의 n-차원 행렬 형식
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
print('전환 전:', type(X_train))

# numpy에서 pandas로 전환
# header 정보 복구 필요
X_train = pd.DataFrame(X_train, columns=names)
X_test = pd.DataFrame(X_test, columns=names)
print('전환 후:', type(X_train))

# 정규화 된 학습용 데이터 출력
print('\n정규화 된 학습용 데이터 샘플 10개')
print(X_train.head(10))
print('정규화 된 학습용 데이터 통계')
print(X_train.describe())

# 학습용 데이터 상자 그림
sns.set(font_scale=2)
sns.boxplot(data=X_train, palette="colorblind")
# plt.show()

    ####### 인공 신경망 구현 #######


# 파이토치 DNN을 Sequential 모델로 구현
model = nn.Sequential(
    nn.Linear(INPUT_DIM, MY_HIDDEN),
    nn.Tanh(),
    nn.Linear(MY_HIDDEN, MY_HIDDEN),
    nn.Tanh(),
    nn.Linear(MY_HIDDEN, 1),
    nn.Sigmoid()
)

print('\nDNN 요약')
print(model)

# 총 파라미터 수 계산
total = sum(p.numel() for p in model.parameters())
print('총 파라미터 수: {:,}'.format(total))