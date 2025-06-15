from flask import Flask, request, jsonify
import joblib

# تحميل النماذج والمعالج
xgb_model = joblib.load('xgb_model.pkl')
lgbm_model = joblib.load('lgbm_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return "🚗 Car Price Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    categorical = [
        data['brand'], data['fuel_type'], data['transmission'],
        data['accident'], data['clean_title']
    ]

    numerical = [
        data['car_age'], data['milage_log'], data['engine_size'],
        data['mileage_per_year'], data['price_per_mile'],
        data['age_mileage_interaction'], data['model_freq']
    ]

    input_data = [categorical + numerical]
    X_processed = preprocessor.transform(input_data)

    xgb_pred = xgb_model.predict(X_processed)[0]
    lgbm_pred = lgbm_model.predict(X_processed)[0]

    return jsonify({
        'xgb_price': round(xgb_pred, 2),
        'lgbm_price': round(lgbm_pred, 2)
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
