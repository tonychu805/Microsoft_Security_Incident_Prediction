# Incident Analysis with TensorFlow

This repository is meant for the [Kaggle project - Microsoft Security Incidnet Prediction](https://www.kaggle.com/datasets/Microsoft/microsoft-security-incident-prediction) project.

### Project Overview:

Microsoft has challenged the data science community to develop methods for predicting significant cybersecurity incidents using GUIDE, the largest public dataset of real-world cybersecurity events. GUIDE includes over 13 million data points across 33 entity types, covering 1.6 million alerts and 1 million annotated incidents with customer-provided triage labels. 

The repository offers a Jupyter notebook and Python scripts for data preprocessing, model development, cross-validation, and evaluation.

### Dataset Description

The dataset is split into two parts:

Raw data from [Kaggle](https://www.kaggle.com/datasets/Microsoft/microsoft-security-incident-prediction/data):
-   GUIDE_Train.csv: Used for training the model.
-   GUIDE_Test.csv: Used for evaluating the model’s performance.

### Objectives

 The goal is to build a predictive model for predicting cybersecurity incident outcomes. The target value for this dataset is called "IncidentGrade", which contains values true positive (TP), benign positive (BP), and false positive (FP)—based on historical data. 

### Featured Methodologies:

Feature Selection: Chi-square test, Anova test, mutual information
Feature engineering: Target encoding, standardization
Machine learning: Random Forest, Gradient Boost, XGBoost, Light Gradient Boost, Catboost 
Validation: K-fold validation

### Results

Due to the size of the datasets, I randomly sampled 80,000 entries and 20,000 entries from both datasets respectedly to achieve an 80:20 train/test ratio.

Model Performances:

| Model          | Precision    | Recall       | F1-Score     | Support     | Accuracy    |
|---------------|--------------|--------------|--------------|-------------|-------------|
| RandomForest  | 0.686257     | 0.67100      | 0.649266     | 20000.0     | 0.67100     |
| GradientBoost | 0.699903     | 0.66380      | 0.624061     | 20000.0     | 0.66380     |
| XGBoost       | 0.703958     | 0.67885      | 0.669383     | 20000.0     | 0.67885     |
| LightGBM      | 0.724249     | 0.69375      | 0.669625     | 20000.0     | 0.69375     |
| CatBoost      | 0.701282     | 0.68900      | 0.669541     | 20000.0     | 0.68900     |

-   LightGBM stands out as the best model overall, with the highest precision, recall, and accuracy. This makes it highly effective in both minimizing false positives and capturing true positives.
-   XGBoost shows strong performance across all metrics, with an F1-score close to LightGBM and CatBoost, making it a strong contender.
-   RandomForest and GradientBoost have lower precision, recall, and F1-scores, indicating they may not perform as well as LightGBM, XGBoost, and CatBoost for this specific task.

Cross Validation (5 folds):

| Model         | 0         | 1         | 2         | 3         | 4         | Mean Accuracy | Std Dev    |
|---------------|-----------|-----------|-----------|-----------|-----------|---------------|------------|
| RandomForest  | 0.699313  | 0.698187  | 0.704750  | 0.701688  | 0.703750  | 0.701537      | 0.002506   |
| GradientBoost | 0.658563  | 0.651375  | 0.656625  | 0.658312  | 0.654062  | 0.655788      | 0.002728   |
| XGBoost       | 0.701500  | 0.700125  | 0.700000  | 0.701313  | 0.698812  | 0.700350      | 0.000978   |
| LightGBM      | 0.696313  | 0.695375  | 0.696375  | 0.698688  | 0.694688  | 0.696288      | 0.001354   |
| CatBoost      | 0.699063  | 0.696313  | 0.698750  | 0.699313  | 0.697187  | 0.698125      | 0.001170   |

-   XGBoost emerges as the most stable model, with the lowest standard deviation across all folds. This means that you can expect it to perform consistently across various datasets, reducing the risk of large fluctuations in performance.
-   RandomForest surprisingly shows the best mean accuracy, but the relatively higher variability suggests that its performance could fluctuate more than XGBoost and CatBoost.
-   CatBoost and LightGBM perform consistently well, with similar mean accuracy and low variability. These models are highly dependable for this task, given their balanced performance.

### Implication

-   LightGBM and XGBoost are the two standout models for predicting cybersecurity incident outcomes given the datsets. LightGBM excels in precision and recall, making it highly effective for reducing false positives and detecting true incidents, while XGBoost is the most consistent performer across different datasets.
-   CatBoost is a strong alternative that balances high accuracy and low variability.
-   RandomForest and GradientBoost could be considered for specific scenarios, but they are less suited for tasks requiring a balance between high precision and recall.

### Acknowledgements

This dataset is hosted by Microsoft Security AI Research.

### Microsoft contacts

-   Scott Freitas (scottfreitas@microsoft.com)
-   Jovan Kalajdjieski (jovank@microsoft.com)
-   Amir Gharib (agharib@microsoft.com)
-   Rob McCann (robmccan@microsoft.com)

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