import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

data = pd.read_csv('climate_nasa.csv')

print(data.info())
print(data.describe())

data['commentsCount'].fillna(0, inplace=True)
data['text'].fillna("No comment", inplace=True)

data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

data['sentiment'] = data['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
print(data[['text', 'sentiment']].head())

vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
text_features = vectorizer.fit_transform(data['text'])
nmf = NMF(n_components=5, random_state=42)
topics = nmf.fit_transform(text_features)

for idx, topic in enumerate(nmf.components_):
    print(f"Topic {idx + 1}: ", [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]])

topic_df = pd.DataFrame(topics, columns=[f"topic_{i+1}" for i in range(topics.shape[1])], index=data.index)
data = pd.concat([data, topic_df], axis=1)

data['year'] = data.index.year
data['month'] = data.index.month
data['day'] = data.index.day

X = data.drop(['likesCount', 'text', 'profileName'], axis=1)
y = data['likesCount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R2 Score: {r2}")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual Likes")
plt.ylabel("Predicted Likes")
plt.title("Actual vs Predicted Likes")
plt.show()

data['likesCount'].resample('Y').mean().plot(kind='line', title="Average Likes Over Years")
plt.xlabel("Year")
plt.ylabel("Average Likes")
plt.show()

future_data = pd.DataFrame({
    'commentsCount': [5, 10, 15],
    'sentiment': [0.2, -0.1, 0.5],
    'year': [2024, 2025, 2026],
    'month': [1, 2, 3],
    'day': [15, 16, 17],
    'topic_1': [0.1, 0.2, 0.3],
    'topic_2': [0.3, 0.1, 0.4],
    'topic_3': [0.2, 0.5, 0.1],
    'topic_4': [0.0, 0.0, 0.1],
    'topic_5': [0.1, 0.1, 0.2],
})

for col in X.columns:
    if col not in future_data.columns:
        future_data[col] = 0
future_data = future_data[X.columns]

future_data_scaled = scaler.transform(future_data)
future_predictions = model.predict(future_data_scaled)
print("Future Predictions (Mock Data):", future_predictions)
