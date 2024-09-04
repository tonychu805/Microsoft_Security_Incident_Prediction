# Incident Analysis with TensorFlow

This repository is meant for the [Kaggle project - Microsoft Security Incidnet Prediction](https://www.kaggle.com/datasets/Microsoft/microsoft-security-incident-prediction) project.

### Project Overview:

Microsoft is challenging the data science community to develop innovative methods for predicting significant cybersecurity incidents. GUIDE, the largest publicly available dataset of real-world cybersecurity incidents, offers researchers and practitioners the opportunity to experiment with authentic data, pushing the boundaries of cybersecurity advancements. This comprehensive dataset includes over 13 million evidence points spanning 33 entity types, covering 1.6 million alerts and 1 million annotated incidents, complete with triage labels from customers over a two-week period. With the release of GUIDE, the goal is to establish a standardized benchmark for guided response systems using real-world data. The dataset’s primary aim is to accurately predict incident triage outcomes—true positive (TP), benign positive (BP), and false positive (FP)—based on historical customer responses.

The repository contains a Jupyter notebook (incident_analysis.ipynb), as well as python scripts that provide a comprehensive guide to analyzing Microsoft's security incident dataset using various machine learning techniques. The notebook covers data preprocessing, model development, cross-validation, and evaluation, providing a step-by-step approach to building robust predictive models.

### Dataset Description

The dataset is split into two parts:

Raw data from [Kaggle](https://www.kaggle.com/datasets/Microsoft/microsoft-security-incident-prediction/data):
- GUIDE_Train.csv: Used for training the model.
- GUIDE_Test.csv: Used for evaluating the model’s performance.

Objectives
Comprehend the limited distribution of the "small" dataset at our disposal.
Generate a sub-dataframe with a 50/50 ratio of "Fraud" and "Non-Fraud" transactions using the NearMiss Algorithm.
Identify the classifiers to be employed and evaluate their respective accuracies, selecting the one with the highest performance.
Summary:
The transaction amounts are relatively modest, with an approximate mean of USD 88 across all transactions.
No "Null" values are present, eliminating the need to devise methods for value replacement.
The majority of transactions (99.83%) are classified as Non-Fraudulent, with Fraudulent transactions occurring only 0.17% of the time in the dataframe.
Feature Technicalities:
PCA Transformation: The data description indicates that all features underwent PCA transformation, a dimensionality reduction technique, except for time and amount.
Scaling: It's important to note that for implementing PCA transformation, features must be scaled beforehand. In this instance, we assume that the dataset developers have scaled all the V features, although this is not explicitly stated.
Results
Accuracy on Logistic Regression Training data : 96.2%
Accuracy on SVC Training data : 96.6%
Accuracy on Decision Tree Training data : 93.6%

###Acknowledgements

This dataset is hosted by Microsoft Security AI Research.

### Microsoft contacts

    Scott Freitas (scottfreitas@microsoft.com)
    Jovan Kalajdjieski (jovank@microsoft.com)
    Amir Gharib (agharib@microsoft.com)
    Rob McCann (robmccan@microsoft.com)

### Citation

    Scott Freitas, Jovan Kalajdjieski, Amir Gharib and Rob McCann. "AI-Driven Guided Response for Security Operation Centers with Microsoft Copilot for Security." arXiv preprint arXiv:2407.09017 (2024).
'
@article{freitas2024ai,
title={AI-Driven Guided Response for Security Operation Centers with Microsoft Copilot for Security},
author={Freitas, Scott and Kalajdjieski, Jovan and Gharib, Amir and McCann, Rob},
journal={arXiv preprint arXiv:2407.09017},
year={2024}
}

### License

    Microsoft is releasing this dataset under the Community Data License Agreement – Permissive – Version 2.0 (CDLA-Permissive-2.0).