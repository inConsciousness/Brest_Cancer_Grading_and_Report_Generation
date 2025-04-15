import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense

df = pd.read_csv('./data/breast-cancer.csv')
df = df.drop(['id', 'Unnamed: 32'], axis=1)
df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1}).astype(int)

X = df.iloc[:, 1:]
y = df.iloc[:, 0]

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build ANN model
model = Sequential()
model.add(Dense(16, activation='relu', input_dim=X.shape[1]))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile & Train
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_split=0.1, batch_size=10, epochs=50)

# Save model
save_model(model, './ann_model/saved_model/model_ann.h5')
print("✅ Model trained and saved at: ./ann_model/saved_model/model_ann.h5")
