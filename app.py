from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

model = pickle.load(open('Rainfall.pkl', 'rb'))
model_columns = pickle.load(open('model_columns.pkl', 'rb'))

# 🔹 Landing Page
@app.route('/')
def landing():
    return render_template('landing.html')

# 🔹 Prediction Form Page
@app.route('/predict_page')
def predict_page():
    return render_template('index.html')

# 🔹 Prediction Logic
@app.route('/predict', methods=['POST'])
def predict():

    input_data = request.form.to_dict()

    numeric_cols = [
        'MinTemp','MaxTemp','Rainfall','Windgustspeed',
        'Windspeed9am','Windspeed3pm','Humidity9am',
        'Humidity3pm','Pressure9am','Pressure3pm',
        'Temp9am','Temp3pm','year','month','day'
    ]

    for col in numeric_cols:
        if col in input_data:
            input_data[col] = float(input_data[col])

    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)
    rain_percent = round(probability[0][1] * 100, 2)

    if prediction[0] == 1:
        return render_template('chance.html', prob=rain_percent)
    else:
        return render_template('noChance.html', prob=rain_percent)

if __name__ == "__main__":
    app.run(debug=True)
