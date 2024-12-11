import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def clean_sm(x): #Convert 1 to 1, and everything else to 0.
    return np.where(x == 1, 1, 0)

def predict_linkedin_usage(model, income, education, parent, married, female, age):
    input_features = pd.DataFrame(
        [[income, education, parent, married, female, age]],
        columns=['income', 'educ2', 'parent', 'married', 'female', 'age']
    )
    # Predict the probability of LinkedIn usage
    probability = model.predict_proba(input_features)[0][1]
    return probability

file_path = 'social_media_usage.csv'
s = pd.read_csv(file_path)

ss = s.copy()
ss['sm_li'] = clean_sm(ss['web1h'])

#clean the data
ss['income'] = ss['income'].apply(lambda x: x if x in range(1, 10) else np.nan)
ss['educ2'] = ss['educ2'].apply(lambda x: x if x in range(1, 9) else np.nan)
ss['parent'] = clean_sm(ss['par'])
ss['married'] = clean_sm(ss['marital'])
ss['female'] = ss['gender'].apply(lambda x: 1 if x == 2 else 0)
ss['age'] = ss['age'].apply(lambda x: x if x <= 98 else np.nan)
ss = ss[['sm_li', 'income', 'educ2', 'parent', 'married', 'female', 'age']]

# Drop rows with any missing values
ss = ss.dropna()

# Create the target vector (y) and feature set (X)
y = ss['sm_li']  # Target variable
X = ss[['income', 'educ2', 'parent', 'married', 'female', 'age']]  # Features

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

log_reg = LogisticRegression(class_weight='balanced', random_state=42)
log_reg.fit(X_train, y_train)

prob_42 = predict_linkedin_usage(log_reg, income=8, education=7, parent=0, married=1, female=1, age=42)
print(f"Probability of LinkedIn usage (42 years old): {prob_42:.2f}")