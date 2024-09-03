import pandas as pd
from scipy.stats import chi2_contingency, f_oneway
from sklearn.feature_selection import mutual_info_classif


# Chi-Square Test for Categorical Features
def chi_square_test(df, feature, target):
    contingency_table = pd.crosstab(df[feature], df[target])
    chi2, p, dof, ex = chi2_contingency(contingency_table)
    return p

# ANOVA Test for Numerical Features
def anova_test(df, feature, target):
    groups = df.groupby(target)[feature].apply(list)
    f_stat, p_value = f_oneway(*groups)
    return p_value

# Mutual Information for Numerical Features
def mutual_information(df, feature, target):
    mi = mutual_info_classif(df[[feature]], df[target])
    return mi[0]




def insignificant_feature(alpha, df):

    chi_square_results = {}
    anova_results = {}
    mutual_info_results = {}

    insignificant_features = []


    for feature in df.columns:
        if feature != 'IncidentGrade':
            if isinstance(df[feature].dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(df[feature]):
                p_value = chi_square_test(df, feature, 'IncidentGrade')
                chi_square_results[feature] = p_value
                if p_value > alpha:
                    insignificant_features.append((feature, 'Chi-Square', p_value))
            elif pd.api.types.is_numeric_dtype(df[feature]):
                p_value = anova_test(df, feature, 'IncidentGrade')
                mi_score = mutual_information(df, feature, 'IncidentGrade')
                anova_results[feature] = p_value
                mutual_info_results[feature] = mi_score
                if p_value > alpha:
                    insignificant_features.append((feature, 'ANOVA', p_value))

    # Sort results
    #sorted_chi_square_results = sorted(chi_square_results.items(), key=lambda item: item[1], reverse=True)
    #sorted_anova_results = sorted(anova_results.items(), key=lambda item: item[1], reverse=True)
    #sorted_mi_results = sorted(mutual_info_results.items(), key=lambda item: item[1], reverse=True)

    # Print insignificant features
    if insignificant_features:
        print("Insignificant Features:")
        for feature, test, p_value in insignificant_features:
            print(f"Feature: {feature}, Test: {test}, P-Value: {p_value}, mutual info: {mi_score}")
    else:
        print("All features are significant at the alpha level of", alpha)