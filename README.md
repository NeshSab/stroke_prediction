# Stroke Prediction: Machine Learning Modeling
Link to the web app:

## Table of contents
- [Introduction](#introduction)
- [Data](#data)
- [Main Findings](#main-findings)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Further Improvements](#further-improvements)
- [Get Help](#get-help)
- [Contribution](#contribution)
<br>

## Introduction
Stroke is one of the leading causes of disability and death worldwide. This project aims to develop a machine learning model to predict stroke occurrences using patient demographic, medical, and lifestyle data.

### Project Goals
- Understand the data & perform EDA.
- Engineer meaningful features while avoiding data leakage.
- Develop & compare tree-based classifiers (Random Forest, XGBoost, LightGBM, Decision Trees).
- Handle imbalanced data appropriately using threshold tuning, feature selection, and scoring metrics.
- Select and deploy the best-performing model.

### Workflow
    - Data Cleaning & Preprocessing
    - Exploratory Data Analysis (EDA)
    - Feature Engineering
    - Model Selection & Hyperparameter Tuning
    - Evaluation & Optimization
<br>

## Data
- **Data Source:** Stroke Prediction Dataset
- **Location:** https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/data
- Dataset overview:
    - 5110 patient records, 80% train, 20% test.
    - Highly imbalanced: 4.87% stroke cases, 95.13% non-stroke cases.
    - Numerical & categorical features:
        - Demographic: Age, gender, marital status, residence type.
        - Medical History: Hypertension, heart disease.
        - Lifestyle: Smoking status, work type.
        - Health Indicators: BMI, glucose levels.

- Potential data leakage considerations:
    - avg_glucose_level: If recorded after a stroke event, it might not be predictive.
    - hypertension & heart_disease: No clear timestamp on when these were recorded relative to stroke occurrence.
    - smoking_status ("Unknown"): Potential missing not at random (MNAR), requires careful interpretation.
<br>

## Main Findings
### Exploratory Data Analysis (EDA)
- No missing values except in bmi (handled via imputation).
- Converted residence_type into a binary feature (urban_resident).
- Outliers in bmi and avg_glucose_level were not removed but flagged with a new feature (glucose_outlier).
- Kept smoking_status = "Unknown" to retain potentially useful information.
- No duplicate entries.

### Models Tested
- Random Forest
- XGBoost
- LightGBM
- Bagging with Decision Tree
- Decision Tree

### Key Experiments and Results
- What improved performance:
    - Using F1-score instead of Precision-Recall for model evaluation;
    - Manually setting probability threshold to 0.3 instead of 0.5;
    - Filling missing values with median rather than -999;
    - Marking missing BMI values with an additional binary feature (bmi_missing).

- What didn’t improve performance:
    - RepeatedStratifiedKFold (increased runtime without better results);
    - SMOTE/SMOTEENN (caused overfitting);
    - GridSearchCV (similar results to RandomizedSearchCV, but slower);
    - Feature Engineering beyond basic transformations (caused multicollinearity).

#### Best Model
- LightGBM
    - Key feature: Age was the most predictive feature.
    - Threshold Optimization: Set to 0.3 for better recall trade-off.
    - Feature Selection: Original features (with additional bmi missing marker) worked best.
<br>

## How to Run
1. Clone the repository from GitHub.
2. Navigate to the project's directory.
3. Install dependencies by running `pip install -r requirements.txt`
4. Run 01_eda.ipynb notebook first in order to create test.csv and train.csv, then 02_modeling.ipynb.
    - Note on model selection: "Training and Hypertuning" script is commented out and data from the final run is saved and loaded. In case there is need to rerun training and evaluation of all models, uncomment the sell.
<br>

- Note on Plotly Plots:
    - Plotly charts do not upload to GitHub as interactive visualizations.
    - To work with interactive charts in a local Notebook environment, set PLOTLY_MODE to "notebook" in data_viz_func.py.
    - Before pushing code to GitHub, revert PLOTLY_MODE to "github" and re-run the analysis notebook to generate static images. <br>
<br>
   
## Project Structure              
- main.ipynb
- 01_eda.ipynb: Core notebook for exploratory data analysis and statistical inference.                  
- 02_modeling.ipynb: Notebook for the feature engineering, comparison of different types of models, best model selection.
- lightgbm.joblib: Saved best model.
- stroke_prediction_app.py: Streamlit script for web application.
- utils folder:
    - stats_utils.py: Custom functions for statistical analysis.
    - vizualization_utils.py: Custom functions for data plotting.
    - eda_utils.py: Custom functions for data processing and analysis, mostly using Pandas.
    - ml_utils.py: Custon functions for machine learning steps from data preprocessing to model evaluation. 
- data folder: Contains saved parameters and results of hypertuning and cross validation.
- plotly_charts folder: Contains static PNG images of Plotly charts for non-interactive environments.
- requirements.txt: Project dependencies.
- README.md: Documentation file describing the whole project and technical aspects of it.
- .flake8: To ignore some of the FLAKE8 features that are in conflict with BLACK formatter.
<br>

## Further Improvements 
- TBD;
<br>

## Get Help
If you encounter any issues or have questions about this project, feel free to reach out. Here are the ways you can get help:
- Open an Issue: if you find a bug, experience problems, or have a feature request, please open an issue.
- Email Me: For personal or specific questions, you can reach me via email at: agneska.sablovskaja@gmail.com.
<br>

## Contribution
Contributions are welcome and appreciated! If you'd like to contribute to this project, here’s how you can get involved:
1. Reporting Bugs: if you find a bug, please open an issue and provide detailed information about the problem. Include steps to reproduce the bug, any relevant logs or error messages, and your environment details (OS, versions, etc.).
2. Suggesting Enhancements: if you have ideas for new features or improvements, feel free to open an issue to discuss it. Describe your suggestion in detail and provide context for why it would be useful.
