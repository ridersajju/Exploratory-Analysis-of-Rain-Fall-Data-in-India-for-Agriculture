import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("Loading dataset...")

df = pd.read_csv("WeatherAUS_cleaned.csv")

# Drop Date if exists
if "Date" in df.columns:
    df.drop("Date", axis=1, inplace=True)

# Fill missing values
for col in df.select_dtypes(include=['float64','int64']).columns:
    df[col].fillna(df[col].mean(), inplace=True)

for col in df.select_dtypes(include=['object']).columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Encode target
le = LabelEncoder()
df["RainTomorrow"] = le.fit_transform(df["RainTomorrow"])

# One-hot encode
df = pd.get_dummies(df, drop_first=True)

# Split data
X = df.drop("RainTomorrow", axis=1)
y = df["RainTomorrow"]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")

model = xgb.XGBClassifier(
    n_estimators=80,
    max_depth=6,
    learning_rate=0.1,
    n_jobs=-1,
    eval_metric='logloss'
)

model.fit(x_train, y_train)

pred = model.predict(x_test)
accuracy = accuracy_score(y_test, pred)

print(f"Accuracy: {accuracy:.4f}")

# Save model & columns
pickle.dump(model, open("Rainfall.pkl", "wb"))
pickle.dump(X.columns, open("model_columns.pkl", "wb"))

print("✓ Model saved successfully")
