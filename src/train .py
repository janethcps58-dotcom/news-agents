
import joblib
from sklearn.linear_model import LogisticRegression

def train(X_train, y_train, model_path="model.pkl"):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    print("Modelo entrenado y guardado en", model_path)

