import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import threading
import os
from datetime import datetime

def create_dashboard(log_file_path):
    app = dash.Dash(__name__)
    
    app.layout = html.Div([
        html.H1('Driver Drowsiness Monitoring Dashboard'),
        dcc.Interval(id='interval-component', interval=5000, n_intervals=0),
        html.Div([
            dcc.Graph(id='live-blinks-graph'),
            dcc.Graph(id='ear-mar-graph'),
            dcc.Graph(id='status-pie')
        ])
    ])
    
    @app.callback(
        [Output('live-blinks-graph', 'figure'),
         Output('ear-mar-graph', 'figure'),
         Output('status-pie', 'figure')],
        [Input('interval-component', 'n_intervals')]
    )
    def update_graphs(n):
        try:
            if not os.path.exists(log_file_path):
                return {}, {}, {}
            
            df = pd.read_csv(log_file_path)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            
            # Blinks Graph
            blinks_fig = go.Figure()
            blinks_fig.add_trace(go.Scatter(
                x=df['Timestamp'],
                y=df['Blink_Count'],
                mode='lines+markers',
                name='Blinks'
            ))
            blinks_fig.update_layout(title='Blink Count Over Time')
            
            # EAR and MAR Graph
            ear_mar_fig = go.Figure()
            ear_mar_fig.add_trace(go.Scatter(
                x=df['Timestamp'],
                y=df['Eye_Aspect_Ratio'],
                mode='lines',
                name='Eye Aspect Ratio'
            ))
            ear_mar_fig.add_trace(go.Scatter(
                x=df['Timestamp'],
                y=df['Mouth_Aspect_Ratio'],
                mode='lines',
                name='Mouth Aspect Ratio'
            ))
            ear_mar_fig.update_layout(title='EAR and MAR Over Time')
            
            # Status Pie Chart
            status_counts = df['Status'].value_counts()
            status_fig = go.Figure(data=[go.Pie(
                labels=status_counts.index,
                values=status_counts.values
            )])
            status_fig.update_layout(title='Status Distribution')
            
            return blinks_fig, ear_mar_fig, status_fig
        except Exception as e:
            print(f"Dashboard update error: {e}")
            return {}, {}, {}
    
    return app

def run_dashboard(log_file_path):
    app = create_dashboard(log_file_path)
    app.run_server(debug=False, port=8050)

def start_dashboard(log_file_path):
    dashboard_thread = threading.Thread(target=run_dashboard, args=(log_file_path,), daemon=True)
    dashboard_thread.start()
    print("Dashboard started at http://localhost:8050")
    return dashboard_thread

if __name__ == '__main__':
    # For standalone testing
    log_file = f"logs/drowsiness_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    start_dashboard(log_file)
