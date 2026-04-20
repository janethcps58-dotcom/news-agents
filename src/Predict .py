

import joblib

def predict(new_data, model_path="model.pkl"):
    model = joblib.load(model_path)
    prediction = model.predict([new_data])
    print("Predicción:", prediction)
