import pickle
from fastapi import FastAPI
from pydantic import BaseModel

# -----------------------
# Load your model + DictVectorizer
# -----------------------
input_file = "pipeline_v1.bin"
with open(input_file, "rb") as f_in:
    dv, model = pickle.load(f_in)

# -----------------------
# Define API + schema
# -----------------------
app = FastAPI(title="Churn Prediction API")

class Customer(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

@app.post("/predict")
def predict(customer: Customer):
    customer_dict = customer.dict()  # Convert to regular Python dict
    X = dv.transform([customer_dict])
    y_pred = model.predict_proba(X)[0, 1]
    return {"churn_probability": float(y_pred)}
