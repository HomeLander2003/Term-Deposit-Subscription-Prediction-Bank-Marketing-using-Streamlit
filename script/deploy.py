class deployment(ML):     
                   
    def deploy(self):
        try:
            if not hasattr(self, 'model_trained') or not self.model_trained:
                st.warning("Please run the ML model before deploying.")
                return

            st.title("Deployment")
            if st.button("Deploy"):
                dump(self.selected_model, "deploy_bank.joblib")
                st.success("Deployment Successful")

        except Exception as e:
            st.warning(e)
