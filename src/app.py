from utils import db_connect
engine = db_connect()

# your code here
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

df_raw = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/linear-regression-project-tutorial/main/medical_insurance_cost.csv')

df_raw.info()

df_raw.sample(10)

df_raw.sex.value_counts()

df_raw.smoker.value_counts()

df_raw.region.value_counts()

df_raw['sex_female']=df_raw['sex'].apply(lambda x:1 if x=='female' else 0)
df_raw['smoker_yes']=df_raw['smoker'].apply(lambda x:1 if x=="yes" else 0)
df_raw['region_sw']=df_raw['region'].apply(lambda x:1 if x=="southwest" else 0)
df_raw['region_nw']=df_raw['region'].apply(lambda x:1 if x=="northwest" else 0)
df_raw['region_ne']=df_raw['region'].apply(lambda x:1 if x=="northeast" else 0)
df_raw=df_raw.drop(['sex','smoker','region'],axis=1)
df_interim=df_raw.copy()

df_interim.head(10)

X=df_interim[['age','bmi','children','sex_female','smoker_yes','region_sw','region_nw','region_ne']]
y=df_interim['charges']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=40)

X_train.corr()

X_train.corr().style.background_gradient(cmap='Blues')

sklm=LinearRegression()
sklm.fit(X_train,y_train)
score=sklm.score(X_train,y_train)
print(f'score is{score: .4f}')

predictions=sklm.predict(X_test)
print(f'The R2 score is:{r2_score(y_test,predictions)}')

y_val_pred=sklm.predict(X_test)
y_train_pred=sklm.predict(X_train)

RMSE_train=mean_squared_error(y_train, y_train_pred, squared=False)
RMSE_test=mean_squared_error(y_test, y_val_pred, squared=False)
print('train :', RMSE_train, 'validation:', RMSE_test)