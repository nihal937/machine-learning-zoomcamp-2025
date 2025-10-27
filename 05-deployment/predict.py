import pickle
from flask import Flask, request, jsonify



C = 1.0
input_file = f'pipeline_v1.bin'
with open(input_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

#app = Flask('churn')

#@app.route('/predict', methods=['POST'])
def predict(customer = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}):
    #customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]

    return y_pred
print(f"prediction: {predict()}")

#if __name__ == '__main__':
#    app.run(debug = True, host='0.0.0.0', port=9696)





