import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.metrics import  classification_report, accuracy_score,RocCurveDisplay,ConfusionMatrixDisplay,confusion_matrix,roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from joblib import load, dump
import streamlit as st
import logging
import os

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s',
    
)

class EDA:
    def __init__(self):
        self.file = None

    def clean(self):
        filepath = r"bank.csv"

        if os.path.isfile(filepath):
            try:
                self.file = pd.read_csv(filepath)
                st.title("Dataset Preview")
                st.write(self.file.head())

                if self.file.isnull().any(axis=1).sum() > 0:
                    self.file.dropna(inplace=True)
                    st.info("Null values removed.")
                else:
                    st.info("No null values found.")

                if self.file.duplicated().sum() > 0:
                    self.file.drop_duplicates(inplace=True)
                    st.info("Duplicate rows removed.")
                else:
                    st.info("No duplicate values found.")
            except Exception as e:
                logging.error(f"Error reading file: {e}")
        else:
            logging.error("File not found at given path.")
            
            
    def save(self):
        
        if self.file is not None:
            try:
                st.title("Save File")
                st.markdown("<p2 style='color:red;'>Press the given Button Below to Save Clean File </p2>", unsafe_allow_html=True)  
                
                st.markdown("""
                                <style>

                                div.stButton > button:first-child:hover {
                                    background-color: red;
                                    color: white;
                                    transform: scale(1.08);
                                    cursor: pointer;
                                }
                                </style>
                            """, unsafe_allow_html=True)

                if st.button("Click Here"):
                    self.file.to_csv("bank_mod.csv",index=False)
                    st.success("File save Successfully")
                    
            except Exception as e:
                st.error(e)
                
        else:
            st.error("Unable to Save File/File not Found")
            

    def analysis(self):
        if self.file is not None:
            try:
                st.title("📊 Descriptive Statistics")
                st.write(self.file.describe())

                self.gr1 = self.file.groupby("loan")["marital"].value_counts().reset_index(name="count")
                self.gr2 = self.file.groupby("loan")["job"].value_counts().reset_index(name="count")
                self.gr3 = self.file.groupby("housing")["marital"].value_counts().reset_index(name="count")
                self.gr4 = self.file.groupby("housing")["job"].value_counts().reset_index(name="count")
                self.gr5=  self.file.groupby("loan")["loan"].count()
                
                
            except Exception as e:
                logging.error(f"Analysis error: {e}")
                st.error("Failed to perform analysis.")
        else:
            st.error("No file loaded. Please run cleaning first.")

    def visualize(self):
        if self.file is not None:
            def subplot(gr1, gr2, gr3, gr4,gr5):
                
                fig1, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

                sns.barplot(x="loan", y="count", hue="marital", data=gr1, palette="coolwarm", ax=axes[0, 0])
                sns.barplot(x="loan", y="count", hue="job", data=gr2, palette="coolwarm", ax=axes[0, 1])
                sns.barplot(x="housing", y="count", hue="marital", data=gr3, palette="Spectral", ax=axes[1, 0])
                sns.barplot(x="housing", y="count", hue="job", data=gr4, palette="Spectral", ax=axes[1, 1])

                axes[0, 0].set_title("Loan per Marital Status", color="purple")
                axes[0, 1].set_title("Loan per Job Status", color="purple")
                axes[1, 0].set_title("Housing per Marital Status", color="purple")
                axes[1, 1].set_title("Housing per Job Status", color="purple")
                
                st.markdown("<h1 style='text-align: center; color: white;'>📊 Visualizations</h1>",unsafe_allow_html=True)
                col1,col2=st.columns(2)

                with col1:
                    
                    st.subheader("**1. House and Loan**")
                    st.pyplot(fig1)
                    
                with col2:
                    
                    st.subheader("**2. Total loan count**")
                    fig2,ax=plt.subplots(figsize=(10,6))
                    # fig2.patch.set_facecolor('none')
                    ax.pie(self.gr5.values, labels=self.gr5.index, autopct='%1.1f%%',textprops={"color" : "purple"},
                        wedgeprops={'width': 0.4}, startangle=90, pctdistance=0.75)

                    st.pyplot(fig2)
                

            subplot(self.gr1, self.gr2, self.gr3,self.gr4,self.gr5)


class ML(EDA):
    def FE(self):
        if self.file is not None:
            try:
                self.file_encoded = pd.get_dummies(self.file, 
                    columns=["job", "marital", "education", "default", "housing", 
                             "loan", "contact", "month", "poutcome"],drop_first=True,dtype=int)
                
                le=LabelEncoder()
                self.file_encoded['deposit']=le.fit_transform(self.file_encoded['deposit'])
                
                
                st.title("🛠️ Feature Engineered Data")
                st.write(self.file_encoded.head())
                            
                X=self.file_encoded.drop("deposit",axis=1)
                y=self.file_encoded["deposit"]
                
                self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=101)
                
                scaler=StandardScaler()
                self.x_train=scaler.fit_transform(self.x_train)
                self.x_test=scaler.transform(self.x_test)
                
            except Exception as e:
                logging.error(f"Feature engineering error: {e}")
        else:
            st.warning("Data not available for feature engineering.")

                
    def ml(self):
        try:
            self.model_trained = False  # default to False before running
            
            models = {
                "Select": None,
                "Logistic Regression": LogisticRegression(),
                "Random Forest": RandomForestClassifier(),
            }

            st.title("Choose Models")
            selected_model_name = st.selectbox("Choose any one", list(models.keys()))
            self.selected_model = models[selected_model_name]

            if self.selected_model is None:
                st.warning("Please select a valid model.")
                return  # stop here if 'Select' is chosen

            score = cross_validate(self.selected_model, self.x_train, self.y_train,scoring=['accuracy', 'f1', 'precision'], cv=5)
            st.markdown("**Validation result**")
            score = pd.DataFrame(score)
            st.write(score)

            st.markdown("**Validation Mean result**")
            st.write(score.mean())
            
            st.markdown("""
                                <style>

                                div.stButton > button:first-child:hover {
                                    background-color: green;
                                    color: white;
                                    transform: scale(1.1);
                                    cursor: pointer;
                                }
                                </style>
                            """, unsafe_allow_html=True)
            
            st.markdown("<h3 style='color:green;'>Check Evaluation on Test Data </h3>", unsafe_allow_html=True)  

            if st.button("See Result"):
                self.selected_model.fit(self.x_train, self.y_train)
                pred = self.selected_model.predict(self.x_test)

                st.write(f"Accuracy: {accuracy_score(self.y_test, pred):.2f}")
                st.text("Classification Report:")
                st.text(classification_report(self.y_test, pred))

                st.subheader("Confusion Matrix")
                fig3, ax1 = plt.subplots()
                ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(self.y_test, pred)).plot(ax=ax1)
                st.pyplot(fig3)

                if len(set(self.y_test)) == 2:
                    st.subheader("ROC Curve")
                    y_proba = self.selected_model.predict_proba(self.x_test)[:, 1]
                    fpr, tpr, _ = roc_curve(self.y_test, y_proba)
                    fig4, ax2 = plt.subplots()
                    RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax2)
                    st.pyplot(fig4)

                # ✅ Model trained successfully
                self.model_trained = True
                st.session_state.model_trained = True


        except Exception as e:
            st.error(e)
            
class deployment(ML):     

    def check_deploy(self):
        """Check if the ML model is trained before deployment."""
        try:
            if "model_trained" not in st.session_state:
                st.session_state.model_trained = False

            if not st.session_state.model_trained:
                st.warning("Please run the ML model before deploying.")
                return False
            return True
        except Exception as e:
            st.warning(e)
            return False
            
    def deploy(self):
        try:
            if "deploy_done" not in st.session_state:
                st.session_state.deploy_done = False
                
                st.markdown("""
                                <style>

                                div.stButton > button:first-child:hover {
                                    background-color: green;
                                    color: white;
                                    transform: scale(1.1);
                                    cursor: pointer;
                                }
                                </style>
                            """, unsafe_allow_html=True)

            st.title("Deployment")
            if not st.session_state.deploy_done:
                if st.button("Deploy"):
                    dump(self.selected_model, "deploy_bank.joblib")
                    st.session_state.deploy_done = True
                    st.success("Deployment Successful ✅")
            else:
                st.success("Model already deployed!")
        except Exception as e:
            st.warning(e)


        

class stream(deployment):
    
    def run_eda(self):
        self.clean()
        self.save()
        self.analysis()
        self.visualize()
        
    def run_fe_ml(self):
        self.clean()
        self.FE()
        self.ml()
        if self.check_deploy():
            self.deploy()
        
        
    def app(self):

        st.sidebar.title("Model Options")

        options = {
            "EDA ":self.run_eda,
            "FE + ML": self.run_fe_ml
        }

        opt_name = st.sidebar.selectbox("Choose Options", list(options.keys()))
        selected_opt = options[opt_name]
        selected_opt()  



# ---------- RUN THE APP ----------
str = stream()
str.app()
