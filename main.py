import pandas as pd
import numpy as np
from preprocess_data import drop_column_with_over_50_percent_missing, fill_missing_value, timestamp_transform
from feature_selection import chi_square_test, anova_test, mutual_information, insignificant_feature


df_train = pd.read_csv("GUIDE_Train.csv", nrows=80000)
df_test = pd.read_csv("GUIDE_Test.csv", nrows=20000)



df_train = df_train.drop(columns = ["Id","OrgId","IncidentId","AlertId","DetectorId","DeviceId"])
df_test = df_test.drop(columns = ["Id","OrgId","IncidentId","AlertId","DetectorId","DeviceId"])


df_train_drop = drop_column_with_over_50_percent_missing(df_train)
df_test_drop = drop_column_with_over_50_percent_missing(df_test)


fill_missing_value(df_train_drop, 'IncidentGrade')
fill_missing_value(df_test_drop, 'IncidentGrade')


timestamp_transform(df_train_drop)
timestamp_transform(df_test_drop)



# Significance threshold
alpha = 0.05

insignificant_feature(alpha, df_train_drop)