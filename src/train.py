from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import mlflow
import mlflow.sklearn

##Train/Test Split

X = df.drop(columns=["is_high_risk"])
y = df["is_high_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


##MLflow Tracking

mlflow.set_experiment("credit-risk-model")

with mlflow.start_run(run_name="LogisticRegression"):
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)

    mlflow.log_metric("roc_auc", auc)
    mlflow.sklearn.log_model(model, "model")
