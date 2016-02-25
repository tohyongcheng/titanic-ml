import pandas as pd
import numpy as np
from scipy.stats import mode

df = pd.read_csv('data/train.csv')
df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# df.info()
age_mean = df['Age'].mean()
df['Age'] = df['Age'].fillna(age_mean)

mode_embarked = mode(df['Embarked'])[0][0]
df['Embarked'] = df['Embarked'].fillna(mode_embarked)

df['Gender'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)
df['Port'] = df['Embarked'].map({'C':1, 'S':2, 'Q':3}).astype(int)

df = df.drop(['Sex', 'Embarked'], axis=1)

cols = df.columns.tolist()
cols = [cols[1]] + cols[0:1] + cols[2:]
df = df[cols]

df.info()

train_data = df.values
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators = 100)
model = model.fit(train_data[0:,2:],train_data[0:,0])

df_test = pd.read_csv('data/test.csv')

df_test.info()
df_test = df_test.drop(['Name', 'Ticket', 'Cabin'], axis=1)

df_test['Age'] = df_test['Age'].fillna(age_mean)
fare_means = df.pivot_table('Fare', index='Pclass', aggfunc='mean')

df_test['Fare'] = df_test[['Fare', 'Pclass']].apply(lambda x:
	fare_means[x['Pclass']] if pd.isnull(x['Fare'])
	else x['Fare'], axis=1)


df_test['Gender'] = df_test['Sex'].map({'female': 0, 'male': 1}).astype(int)
df_test['Port'] = df_test['Embarked'].map({'C':1, 'S':2, 'Q':3})

df_test = df_test.drop(['Sex', 'Embarked'], axis=1)

test_data = df_test.values

output = model.predict(test_data[:,1:])

result = np.c_[test_data[:,0].astype(int), output.astype(int)]
df_result = pd.DataFrame(result[:,0:2], columns=['PassengerId', 'Survived'])

df_result.to_csv('results/titanic_1-1.csv', index=False)



