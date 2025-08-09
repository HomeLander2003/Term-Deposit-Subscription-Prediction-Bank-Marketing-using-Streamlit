# Term-Deposit-Subscription-Prediction-Bank-Marketing-using-Streamlit

 Streamlit-based end-to-end machine learning application for data cleaning, exploratory data analysis (EDA), feature engineering, model training, evaluation, and deployment using a bank dataset.

Here’s a breakdown of its components and functionality:

1. Libraries Used
Data Handling & Analysis: numpy, pandas

Visualization: matplotlib, seaborn

ML Preprocessing: StandardScaler, LabelEncoder

Model Selection & Evaluation: train_test_split, cross_validate, classification metrics

Models: LogisticRegression, RandomForestClassifier

Model Saving/Loading: joblib

UI: streamlit

Logging & File Handling: logging, os

2. Class Structure
Class: EDA
Handles data cleaning, saving, descriptive analysis, and visualizations.

Methods:
clean()

Loads the CSV file from a specified path (bank.csv)

Removes null values and duplicates (with Streamlit notifications)

Displays the dataset preview

save()

Saves the cleaned data as bank_mod.csv when a button is clicked

analysis()

Generates grouped summaries for combinations of categorical variables (loan, marital, job, housing)

Stores them for later visualization

visualize()

Displays bar plots (loan vs marital/job, housing vs marital/job)

Displays pie chart for total loan count

Class: ML (inherits EDA)
Handles feature engineering, model training, validation, and testing.

Methods:
FE()

Applies One-Hot Encoding to categorical columns

Encodes the target (deposit) with LabelEncoder

Splits the data into train/test sets (70/30)

Scales features using StandardScaler

Lets the user choose between Logistic Regression and Random Forest

Performs 5-fold cross-validation (accuracy, f1, precision) and shows mean scores

On button click, trains the chosen model, evaluates on test data, and shows:

Accuracy

Classification report

Confusion matrix

ROC curve (for binary classification)

Class: deployment (inherits ML)
Handles saving the trained model.

Method:
deploy()

Saves the selected trained model as deploy_bank.joblib for future use.

Class: stream (inherits deployment)
The main application controller for Streamlit UI.

Methods:
run_eda() → Runs the cleaning, saving, analysis, and visualization steps

run_fe_ml() → Runs cleaning, feature engineering, ML training, and deployment

app() → Creates a sidebar menu in Streamlit for user to choose:

EDA

FE + ML
and executes the corresponding workflow

3. Application Flow
Sidebar Menu → User selects EDA or FE + ML

EDA Option:

Load & clean data

Save cleaned file

Perform grouped analysis

Show visualizations

FE + ML Option:

Load & clean data

Feature engineer

Model selection + cross-validation

Test evaluation (classification metrics, confusion matrix, ROC curve)

Deploy model if desired

4. Key Features
✅ Complete ML Pipeline: From raw CSV to deployed model
✅ Interactive UI with Streamlit
✅ EDA Visualizations: Bar plots & pie charts
✅ Automatic Data Cleaning (null & duplicate removal)
✅ Cross-Validation & Final Testing
✅ Model Deployment (joblib format)
✅ Custom Button Styling via inline CSS
