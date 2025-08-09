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
                
                x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
                
                scaler=StandardScaler()
                x_train=scaler.fit_transform(x_train)
                x_test=scaler.transform(x_test)
                
                
                models = {
                    "Select":"Select",
                    "Logistic Regression": LogisticRegression(),
                    "Random Forest": RandomForestClassifier(),
                }

                st.title("Choose Models")
                selected_model_name = st.selectbox("Choose any one", list(models.keys()))

                self.selected_model = models[selected_model_name]
                
                score=cross_validate(self.selected_model,x_train,y_train,scoring=['accuracy','f1','precision'],cv=5)
                st.markdown("**Validation result**")
                score=pd.DataFrame(score)
                st.write(score)
                st.markdown("**Validation Mean result**")
                mean_score = score.mean()
                st.write(mean_score)
                
                st.markdown("<h4 style='text-align: center; color: green;'>If Satisfy with the Validation Click Button to see Final Evaluation on Test Data</h4>"
                    ,unsafe_allow_html=True)
                
                
                st.markdown("""
                                <style>

                                div.stButton > button:first-child:hover {
                                    background-color: green;
                                    color: white;
                                    transform: scale(1.08);
                                    cursor: pointer;
                                }
                                </style>
                            """, unsafe_allow_html=True)
                
                if st.button("See Result"):
                    
                    self.selected_model.fit(x_train, y_train)
                    pred = self.selected_model.predict(x_test)


                    st.write(f"Accuracy: {accuracy_score(y_test, pred):.2f}")
                    report = classification_report(y_test, pred)
                    st.text("Classification Report:")
                    st.text(f"""```
                    {report}
                    ```""")
                    
                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(y_test, pred)
                    fig3, ax1 = plt.subplots()
                    cnf_disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                    cnf_disp.plot(ax=ax1)
                    st.pyplot(fig3)
                    
                    if len(set(y_test)) == 2:
                        st.subheader("ROC Curve")
                        y_proba = self.selected_model.predict_proba(x_test)[:, 1]  # Probabilities for class 1
                        fpr, tpr, _ = roc_curve(y_test, y_proba)
                        fig4, ax2 = plt.subplots()
                        roc_disp = RocCurveDisplay(fpr=fpr, tpr=tpr)
                        roc_disp.plot(ax=ax2)
                        st.pyplot(fig4)
                    else:
                        st.warning("ROC Curve is only available for binary classification problems.")
            
                
            except Exception as e:
                logging.error(f"Feature engineering error: {e}")
        else:
            st.warning("Data not available for feature engineering.")
