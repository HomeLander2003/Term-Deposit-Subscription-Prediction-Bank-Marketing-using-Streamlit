            
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

        except Exception as e:
            st.error(e)
