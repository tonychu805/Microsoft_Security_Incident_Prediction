
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder


def model_comparison(X_train_scaled, X_test_scaled, df_train_Y, df_test_Y):
    results = {}

    # Random Forest
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train_scaled, df_train_Y)
    y_pred_rf = rf_model.predict(X_test_scaled)

    rf_report = classification_report(df_test_Y, y_pred_rf, output_dict=True)
    results['RandomForest'] = rf_report['macro avg']
    results['RandomForest'].update(rf_report['weighted avg'])
    results['RandomForest']['accuracy'] = rf_report['accuracy']

    #print("Random Forest Classifier Report:")
    #print(classification_report(df_test_Y, y_pred_rf))

    # Gradient Boost
    gb_model = GradientBoostingClassifier(random_state=42)
    gb_model.fit(X_train_scaled, df_train_Y)
    y_pred_gb = gb_model.predict(X_test_scaled)

    gb_report = classification_report(df_test_Y, y_pred_gb, output_dict=True)
    results['GradientBoost'] = gb_report['macro avg']
    results['GradientBoost'].update(gb_report['weighted avg'])
    results['GradientBoost']['accuracy'] = gb_report['accuracy']

    #print("Gradient Boosting Classifier Report:")
    #print(classification_report(df_test_Y, y_pred_gb))

    # XGBoost
    xgb_model = XGBClassifier(random_state=42)
    le = LabelEncoder()
    label_train_y = le.fit_transform(df_train_Y)
    label_test_y = le.transform(df_test_Y)

    xgb_model.fit(X_train_scaled, label_train_y)
    y_pred_xgb = xgb_model.predict(X_test_scaled)

    xgb_report = classification_report(label_test_y, y_pred_xgb, output_dict=True)
    results['XGBoost'] = xgb_report['macro avg']
    results['XGBoost'].update(xgb_report['weighted avg'])
    results['XGBoost']['accuracy'] = xgb_report['accuracy']

    #print("XGBoost Classifier Report:")
    #print(classification_report(label_test_y, y_pred_xgb))

    # LightGBM
    lgbm_model = LGBMClassifier(random_state=42)
    lgbm_model.fit(X_train_scaled, df_train_Y)
    y_pred_lgbm = lgbm_model.predict(X_test_scaled)

    lgbm_report = classification_report(df_test_Y, y_pred_lgbm, output_dict=True)
    results['LightGBM'] = lgbm_report['macro avg']
    results['LightGBM'].update(lgbm_report['weighted avg'])
    results['LightGBM']['accuracy'] = lgbm_report['accuracy']

    #print("LightGBM Classifier Report:")
    #print(classification_report(df_test_Y, y_pred_lgbm))

    # CatBoost
    catboost_model = CatBoostClassifier(random_state=42, verbose=0)
    catboost_model.fit(X_train_scaled, df_train_Y)
    y_pred_catboost = catboost_model.predict(X_test_scaled)

    catboost_report = classification_report(df_test_Y, y_pred_catboost, output_dict=True)
    results['CatBoost'] = catboost_report['macro avg']
    results['CatBoost'].update(catboost_report['weighted avg'])
    results['CatBoost']['accuracy'] = catboost_report['accuracy']

    #print("CatBoost Classifier Report:")
    #print(classification_report(df_test_Y, y_pred_catboost))

    # Convert the results dictionary into a DataFrame
    results_df = pd.DataFrame(results)

    # Transpose the DataFrame for better readability
    results_df = results_df.T

    # Display the combined DataFrame
    return results_df




from sklearn.model_selection import cross_val_score, KFold

# Assume X_train_scaled and df_train_Y are already defined
# X_train_scaled: Scaled features for training
# df_train_Y: Target variable for training

# Initialize K-Fold cross-validation with 5 folds


def cross_validation(X_train_scaled, df_train_Y, kf):
    # Initialize a dictionary to hold the cross-validation results
    cv_results = {}

    # Random Forest
    rf_model = RandomForestClassifier(random_state=42)
    rf_scores = cross_val_score(rf_model, X_train_scaled, df_train_Y, cv=kf, scoring='accuracy')
    cv_results['RandomForest'] = rf_scores
    #print(f"Random Forest CV Accuracy: {rf_scores.mean():.4f} (+/- {rf_scores.std():.4f})")

    # Gradient Boosting
    gb_model = GradientBoostingClassifier(random_state=42)
    gb_scores = cross_val_score(gb_model, X_train_scaled, df_train_Y, cv=kf, scoring='accuracy')
    cv_results['GradientBoost'] = gb_scores
    #print(f"Gradient Boosting CV Accuracy: {gb_scores.mean():.4f} (+/- {gb_scores.std():.4f})")

    # XGBoost
    xgb_model = XGBClassifier(random_state=42)
    le = LabelEncoder()
    label_train_y = le.fit_transform(df_train_Y)

    xgb_scores = cross_val_score(xgb_model, X_train_scaled, label_train_y, cv=kf, scoring='accuracy')
    cv_results['XGBoost'] = xgb_scores
    #print(f"XGBoost CV Accuracy: {xgb_scores.mean():.4f} (+/- {xgb_scores.std():.4f})")

    # LightGBM
    lgbm_model = LGBMClassifier(random_state=42)
    lgbm_scores = cross_val_score(lgbm_model, X_train_scaled, df_train_Y, cv=kf, scoring='accuracy')
    cv_results['LightGBM'] = lgbm_scores
    #print(f"LightGBM CV Accuracy: {lgbm_scores.mean():.4f} (+/- {lgbm_scores.std():.4f})")

    # CatBoost
    catboost_model = CatBoostClassifier(random_state=42, verbose=0)
    catboost_scores = cross_val_score(catboost_model, X_train_scaled, df_train_Y, cv=kf, scoring='accuracy')
    cv_results['CatBoost'] = catboost_scores
    #print(f"CatBoost CV Accuracy: {catboost_scores.mean():.4f} (+/- {catboost_scores.std():.4f})")

    # Convert the cross-validation results into a DataFrame for easier comparison
    cv_results_df = pd.DataFrame(cv_results)

    # Transpose the DataFrame for better readability
    cv_results_df = cv_results_df.T

    # Add a column for the mean accuracy and standard deviation
    cv_results_df['Mean Accuracy'] = cv_results_df.mean(axis=1)
    cv_results_df['Std Dev'] = cv_results_df.std(axis=1)

    # Display the combined cross-validation results
    #print(cv_results_df)
    return cv_results_df