# Incident Analysis with TensorFlow

### Data

This repository is dedicated to the [Kaggle project - Microsoft Security Incidnet Prediction](https://www.kaggle.com/datasets/Microsoft/microsoft-security-incident-prediction).

Table of Contents:

	1.	Project Overview
	2.	Dataset Description
	3.	Installation
	4.	Usage
	5.	Notebook Structure
    6.  Environment Setup
	7.	Data Loading and Exploration
	8.	Data Preprocessing
	9.	Feature Engineering
	10.	Model Building
	11.	Model Evaluation
	12.	Results
	13.	License
	14.	Acknowledgments

Project Overview:

Microsoft is challenging the data science community to develop innovative methods for predicting significant cybersecurity incidents. GUIDE, the largest publicly available dataset of real-world cybersecurity incidents, offers researchers and practitioners the opportunity to experiment with authentic data, pushing the boundaries of cybersecurity advancements. This comprehensive dataset includes over 13 million evidence points spanning 33 entity types, covering 1.6 million alerts and 1 million annotated incidents, complete with triage labels from customers over a two-week period. With the release of GUIDE, the goal is to establish a standardized benchmark for guided response systems using real-world data. The dataset’s primary aim is to accurately predict incident triage outcomes—true positive (TP), benign positive (BP), and false positive (FP)—based on historical customer responses.

The repository contains a Jupyter notebook (incident_analysis.ipynb), as well as python scripts that provide a comprehensive guide to analyzing Microsoft's security incident dataset using various machine learning techniques. The notebook covers data preprocessing, model development, cross-validation, and evaluation, providing a step-by-step approach to building robust predictive models.

Dataset Description

The dataset is split into two parts:

	Raw data from [Kaggle](https://www.kaggle.com/datasets/Microsoft/microsoft-security-incident-prediction/data):
    •	GUIDE_Train.csv: Used for training the model.
    •	GUIDE_Test.csv: Used for evaluating the model’s performance.

Repository Structure

1. Environment Setup

    1. Ensure [Anaconda](https://www.anaconda.com/download/) is installed

        conda --version

    2. Locate / download the enviroment.yml file

    3. Create the conda environment from the .yml file:

        conda env create -f environment.yml

    4. Activate the new environment (the name is defined under the name field in the .yml file)

        conda activate your_env_name

    5. Verify the installation

        conda env list



Data Loading and Exploration

	•	Data Loading:
	•	The dataset is loaded using pd.read_csv, with a subset of rows (nrows=10000) used for initial exploration.
	•	Initial Exploration:
	•	The first few rows of the dataset are displayed using df_train.head().
	•	Data types and basic statistics are checked using df_train.info().

Data Preprocessing

	•	Dropping Unnecessary Columns:
	•	Columns such as Id, OrgId, IncidentId, AlertId, DetectorId, DeviceId are dropped as they do not contribute to the predictive model.
	•	Handling Missing Data:
	•	A custom function drop_column_with_over_50_percent_missing(df) is defined to remove columns with more than 50% missing values.
	•	Further Cleaning:
	•	Other preprocessing steps might include filling missing values, normalization, and standardization (though these are inferred from the initial steps and would be detailed in subsequent cells).

Feature Engineering

	•	Feature Selection:
	•	Specific features that contribute to the prediction of the target variable are selected based on exploratory analysis.
	•	Transformation:
	•	Features are transformed as necessary, possibly including encoding categorical variables, scaling numerical features, etc.

Model Building

	•	Model Architecture:
	•	A neural network model is built using TensorFlow’s Sequential API.
	•	The model likely includes multiple layers (Dense layers) to capture the complex relationships in the data.
	•	Compilation:
	•	The model is compiled with appropriate loss functions, optimizers, and evaluation metrics.
	•	Training:
	•	The model is trained on the training dataset, with the process including tracking of loss and accuracy over epochs.

Model Evaluation

	•	Performance Metrics:
	•	The model’s performance is evaluated using various metrics such as accuracy, precision, recall, and F1-score (these would be included based on the specific code cells dedicated to evaluation).
	•	Visualization:
	•	Training and validation loss/accuracy are plotted to assess the model’s learning behavior.
	•	Confusion matrix, ROC curves, or other diagnostic plots might be used to evaluate model performance on the test dataset.

Results

The notebook provides a detailed analysis of the data and the resulting predictive model. The key results include:

	•	A trained TensorFlow model capable of predicting security incidents.
	•	Insights into the most significant features contributing to predictions.
	•	Evaluation metrics that help understand the model’s strengths and areas for improvement.

Acknowledgements
This dataset is hosted by Microsoft Security AI Research.

Microsoft contacts

    Scott Freitas (scottfreitas@microsoft.com)
    Jovan Kalajdjieski (jovank@microsoft.com)
    Amir Gharib (agharib@microsoft.com)
    Rob McCann (robmccan@microsoft.com)

Citation

    Scott Freitas, Jovan Kalajdjieski, Amir Gharib and Rob McCann. "AI-Driven Guided Response for Security Operation Centers with Microsoft Copilot for Security." arXiv preprint arXiv:2407.09017 (2024).
'
@article{freitas2024ai,
title={AI-Driven Guided Response for Security Operation Centers with Microsoft Copilot for Security},
author={Freitas, Scott and Kalajdjieski, Jovan and Gharib, Amir and McCann, Rob},
journal={arXiv preprint arXiv:2407.09017},
year={2024}
}

License

    Microsoft is releasing this dataset under the Community Data License Agreement – Permissive – Version 2.0 (CDLA-Permissive-2.0).