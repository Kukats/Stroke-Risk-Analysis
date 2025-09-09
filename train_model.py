from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from preprocess import load_and_preprocess

# Load data
X_train, X_test, y_train, y_test, feature_names = load_and_preprocess("data/stroke_data_sample.csv")

# Logistic Regression
log_reg = LogisticRegression(max_iter=500, class_weight="balanced")
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)
print("=== Logistic Regression ===")
print(classification_report(y_test, y_pred_lr))

# Decision Tree
dt = DecisionTreeClassifier(max_depth=5, random_state=42, class_weight="balanced")
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print("=== Decision Tree ===")
print(classification_report(y_test, y_pred_dt))
