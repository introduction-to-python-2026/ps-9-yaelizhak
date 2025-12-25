import pandas as pd
df = pd.read_csv('parkinsons.csv')
df.head()

import seaborn as sns
import matplotlib.pyplot as plt
sns.pairplot(df, hue="status", diag_kind="kde", corner=True, palette="viridis")
plt.show()

selected_features = ['PPE',  'DFA']
X = df[selected_features]
y = df['status']

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.svm import SVC 
model = SVC(kernel='rbf', gamma='scale') 
model.fit(X_train, y_train)
predictions = model.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)

import joblib
joblib.dump(model, 'my_model.joblib')


