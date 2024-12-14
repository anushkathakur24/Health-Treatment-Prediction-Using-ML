import os  
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
import seaborn as sns

train_df = pd.read_csv('train_data.csv')
test_df = pd.read_csv('test_data.csv')

print("Current Working Directory:", os.getcwd())
print("Training data loaded successfully!")
print("Test data loaded successfully!")


sns.set(style="whitegrid")

plt.figure(figsize=(8, 5))
sns.countplot(data=train_df, x='Treatment_Type')
plt.title('Distribution of Treatment Types')
plt.xlabel('Treatment Type')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(data=train_df, x='Sex')
plt.title('Count of Sex')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(12, 8))
correlation_matrix = train_df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


label_encoder = LabelEncoder()
train_df['Sex'] = label_encoder.fit_transform(train_df['Sex'].astype(str))
train_df['Treatment_Type'] = label_encoder.fit_transform(train_df['Treatment_Type'].astype(str))

X = train_df.drop(['id', 'Treatment_Type'], axis=1)
y = train_df['Treatment_Type']

X = pd.get_dummies(X, drop_first=True)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best parameters found:", grid_search.best_params_)

best_rf_model = grid_search.best_estimator_
val_predictions = best_rf_model.predict(X_val)
val_accuracy = accuracy_score(y_val, val_predictions)

print("Validation Accuracy:", val_accuracy)

X_test = test_df.drop(['id'], axis=1)

X_test['Sex'] = pd.Categorical(X_test['Sex'], categories=label_encoder.classes_).codes
X_test = pd.get_dummies(X_test, drop_first=True)

X_test = X_test.reindex(columns=X.columns, fill_value=0)

X_test_scaled = scaler.transform(X_test)

predictions = best_rf_model.predict(X_test_scaled)

submission_df = pd.DataFrame({'id': test_df['id'], 'Treatment_Type': predictions})
submission_df.to_csv('submission2.csv', index=False)
print("Submission file created!")
