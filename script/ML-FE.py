class ML():
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
