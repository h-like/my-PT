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
