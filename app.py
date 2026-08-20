import joblib
test= [
    7,
    338,
    "Salaried",
    34,
    "Married",
    2,
    544644,
    0,
    0.0441,
    18015,
    6333,
    30489,
    12,
    "Education",
    "Rural",
    "Not Graduate",
    "nan",
    "MNC"
]
label_encoder = joblib.load("label_encoder.pkl")
onehot_encoder = joblib.load("onehot_encoder.pkl")
scaler = joblib.load("scaler.pkl")
model = joblib.load("logistic_regression_model.pkl")

# 1. Label Encoding
test[-3] = label_encoder.transform(
    [test[-3]]
)
print(test[-3])
# Updated cols list using integer indices instead of string names
cols = [
    2,   # "Employment_Status"
    4,   # "Marital_Status"
    13,  # "Loan_Purpose"
    14,  # "Property_Area"
    16,  # "Gender"
    17   # "Employer_Category"
]

test_values = [[]]
for i in cols:
    test_values[0].append(test[i])
    test[i]=True
print(test)
print(test_values)
#2 Transform using the 2D list
encoded = onehot_encoder.transform(test_values)
print(encoded)
#3  Combine numerical + encoded columns
