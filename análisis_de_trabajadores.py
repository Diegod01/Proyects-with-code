

import pandas as pd

df = pd.read_csv('/content/Social_Network_Ads.csv')

df

df.shape

df.isnull().sum()

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes[0,0].set_title('Age Of People')
sns.histplot(ax=axes[0,1],x='Age',data=df,color="g")
axes[0,1].set_title('Distribution Of Ages')

axes[1,0].set_title('Estimated Salary Of People')
sns.histplot(ax=axes[1,1],x='EstimatedSalary',data=df,color="y")
axes[1,1].set_title('Distribution Of Estimated Salary')
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
sns.histplot(ax=axes[0,1],x='Age',data=df,color="g")
axes[0,1].set_title('Distribution Of Ages')

sns.histplot(ax=axes[1,1],x='EstimatedSalary',data=df,color="y")
axes[1,1].set_title('Distribution Of Estimated Salary')
plt.show()

fig ,axes = plt.subplots(1,2, figsize=(15,5))
sns.countplot(ax=axes[0],x='Purchased',data=df)
axes[0].set_title('Number Of People Purchased')
sns.countplot(ax=axes[1],x='Purchased',hue='Gender',data=df,palette="magma")
axes[1].set_title('Number Of People Purchased By Gender')
plt.show()

df.corr()

import matplotlib.pyplot as plt


corr_df = df.corr(method='pearson')

plt.matshow(corr_df)
plt.show()

f,ax = plt.subplots(figsize=(5, 5))

sns.heatmap(df.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',cmap='viridis',ax=ax)
plt.show()

import seaborn as sns
corr_df = df.corr(method='pearson')

plt.figure(figsize=(8, 6))
sns.heatmap(corr_df, annot=True)
plt.show()

corr_df = df.corr(method='pearson')

corr_df.style.background_gradient(cmap='coolwarm')



df.drop('User ID',axis = 1, inplace = True)
label = {'Male': 0 ,"Female" : 1}
df['Gender'].replace(label, inplace= True)

df.head()

sns.pairplot(df)

df.head()

import pandas as pd
import numpy as np

df.columns.values

print(df[['Gender','Age']])

df_copy = (df[['Gender','Age']])

df_copy

df_copy.shape

X= df_copy['Gender']
y= df_copy['Age']

from sklearn.linear_model import LinearRegression

model = LinearRegression()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

X_train=np.array(X_train)
y_train=np.array(y_train)

model.fit(X_train.reshape(-1,1),y_train.reshape(-1,1))

model.score(X_train.reshape(-1,1),y_train.reshape(-1,1))

y_test=np.array(y_test)
X_test= np.array(X_test)

y_pred = model.predict(X_test.reshape(-1,1))

y_pred

import matplotlib.pyplot as plt

plt.scatter(y_test,y_pred)

#No hay correlación entre el género y la edad

df.columns.values

df_copy2 = (df[['EstimatedSalary','Purchased']])

df_copy2

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(color_codes=True)

df.columns.values

df.columns.values

sns.lmplot(x="EstimatedSalary", y="Age", data=df);





