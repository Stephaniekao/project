from utils import *

# Start the Gradio app
if __name__ == "__main__":
    dashboard = ParisAirbnbDashboard()
    dashboard.launch_app()      
    dashboard.demo.launch()