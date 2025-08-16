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
