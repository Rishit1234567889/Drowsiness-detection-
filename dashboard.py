# dashboard.py
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd

def create_dashboard(log_file):
    app = dash.Dash(__name__)
    
    def load_data():
        return pd.read_csv(log_file)
    
    app.layout = html.Div([
        html.H1("Drowsiness Detection Dashboard"),
        
        dcc.Graph(id='ear-plot'),
        dcc.Graph(id='mar-plot'),
        dcc.Graph(id='status-pie'),
        
        dcc.Interval(
            id='interval-component',
            interval=5*1000,  # Update every 5 seconds
            n_intervals=0
        )
    ])

    @app.callback(
        [Output('ear-plot', 'figure'),
         Output('mar-plot', 'figure'),
         Output('status-pie', 'figure')],
        [Input('interval-component', 'n_intervals')]
    )
    def update_graphs(n):
        df = load_data()
        
        ear_fig = go.Figure()
        ear_fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['Eye_Aspect_Ratio'],
                                   mode='lines', name='EAR'))
        ear_fig.update_layout(title='Eye Aspect Ratio Over Time')
        
        mar_fig = go.Figure()
        mar_fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['Mouth_Aspect_Ratio'],
                                   mode='lines', name='MAR'))
        mar_fig.update_layout(title='Mouth Aspect Ratio Over Time')
        
        status_counts = df['Status'].value_counts()
        status_fig = go.Figure(data=[go.Pie(labels=status_counts.index,
                                          values=status_counts.values)])
        status_fig.update_layout(title='Status Distribution')
        
        return ear_fig, mar_fig, status_fig
    
    return app

if __name__ == '__main__':
    # This will only run if you run dashboard.py directly
    app = create_dashboard('drowsiness_log.csv')  # Replace with your log file path
    app.run_server(debug=True)