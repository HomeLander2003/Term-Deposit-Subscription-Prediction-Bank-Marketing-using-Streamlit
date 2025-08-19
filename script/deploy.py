            
class deployment():     

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
                    
                    with st.spinner("Checking Model..."):
                        time.sleep(1)
                    with st.spinner("Collecting Information..."):
                        time.sleep(1)
                    with st.spinner("Deploying..."):
                        time.sleep(1)
                    
                    dump(self.selected_model, "deploy_bank.joblib")
                    st.session_state.deploy_done = True
                    st.success("Deployment Successful ✅")
            else:
                st.success("Model already deployed!")
        except Exception as e:
            st.warning(e)
