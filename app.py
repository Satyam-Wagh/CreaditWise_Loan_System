import joblib as jb
import warnings

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names"
)
#step 1:import models
scaler = jb.load("scaler.pkl")
model=jb.load("logistic_regression_model.pkl")
test = [
    322,
    235,
    2345,
    245,
    245,
    245,
    245,
    245,
    245,
    0,
    0,
    1,
    0,
    1,
    1,
    0,
    0,
    0,
    1,
    0,
    1,
    1,
    0,
    0,
    0,
    425,
    245
]
#step 2:scale the values
scale_values = scaler.transform([test])

print(scale_values)
#step 3:Predict the result
result=model.predict(scale_values)
print(result)
