import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from preprocess_data import drop_column_with_over_50_percent_missing, fill_missing_value, timestamp_transform
from feature_selection import chi_square_test, anova_test, mutual_information, insignificant_feature
from feature_transformation import target_encode, scale_data
from model_building_and_validation import model_comparison, cross_validation


# Load data
df_train = pd.read_csv("GUIDE_Train.csv", nrows=80000)
df_test = pd.read_csv("GUIDE_Test.csv", nrows=20000)


# Feature Engineering

### Feature Selection
df_train = df_train.drop(columns = ["Id","OrgId","IncidentId","AlertId","DetectorId","DeviceId"])
df_test = df_test.drop(columns = ["Id","OrgId","IncidentId","AlertId","DetectorId","DeviceId","Usage"])

### Drop features that are missing more than half of the values
df_train_drop = drop_column_with_over_50_percent_missing(df_train)
df_test_drop = drop_column_with_over_50_percent_missing(df_test)

### Fill up missing value using Mode
fill_missing_value(df_train_drop, 'IncidentGrade')
fill_missing_value(df_test_drop, 'IncidentGrade')

### Transform timestamp into datetime
timestamp_transform(df_train_drop)
timestamp_transform(df_test_drop)


### Significance Test 

# Significance threshold
alpha = 0.05

insignificant_feature(alpha, df_train_drop)
insignificant_feature(alpha, df_test_drop)
### Scale

df_train_X = df_train_drop.drop(columns=["IncidentGrade"])
df_train_Y = df_train_drop["IncidentGrade"]
df_test_X = df_test_drop.drop(columns=["IncidentGrade"])
df_test_Y = df_test_drop["IncidentGrade"]

### Feature Encoding

X_train_encoded = target_encode(df_train_X, 100)
X_test_encoded = target_encode(df_test_X, 100)

X_train_scaled, X_test_scaled = scale_data(X_train_encoded, X_test_encoded)

### Model Building

model_performance = model_comparison(X_train_scaled, X_test_scaled, df_train_Y, df_test_Y)

## Cross Validation

kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_results = cross_validation(X_train_scaled, df_train_Y, kf)