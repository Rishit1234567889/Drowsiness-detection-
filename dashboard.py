import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import threading
import os
from datetime import datetime

def calculate_metrics(df):
    """
    Calculate comprehensive metrics from drowsiness data
    
    Returns:
    - Accuracy metrics
    - Yawn intensity distribution
    - Temporal analysis
    """
    if df.empty:
        return {
            'overall_accuracy': 0,
            'status_breakdown': {},
            'yawn_intensity': {},
            'temporal_analysis': {}
        }
    
    # Total records
    total_records = len(df)
    
    # Status breakdown
    status_counts = df['Status'].value_counts()
    status_percentages = (status_counts / total_records * 100).to_dict()
    
    # Accuracy (percentage of 'Active' time)
    active_count = status_counts.get('Active', 0)
    overall_accuracy = (active_count / total_records) * 100
    
    # Yawn intensity analysis
    yawn_statuses = ['Mild Yawn', 'Moderate Yawn !', 'Severe Yawn !!!']
    yawn_counts = {status: len(df[df['Status'] == status]) for status in yawn_statuses}
    
    # Temporal analysis
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    temporal_analysis = {
        'total_duration': (df['Timestamp'].max() - df['Timestamp'].min()).total_seconds(),
        'avg_blinks_per_minute': df['Blink_Count'].mean(),
        'max_consecutive_active_time': None,  # Could be calculated with more complex logic
    }
    
    return {
        'overall_accuracy': round(overall_accuracy, 2),
        'status_breakdown': status_percentages,
        'yawn_intensity': yawn_counts,
        'temporal_analysis': temporal_analysis
    }

def create_dashboard(log_file_path):
    app = dash.Dash(__name__)
    
    app.layout = html.Div([
        html.H1('Driver Drowsiness Monitoring Dashboard', 
                style={'textAlign': 'center', 'color': '#333'}),
        
        dcc.Interval(id='interval-component', interval=5000, n_intervals=0),
        
        html.Div([
            # Metrics and Summary
            html.Div([
                html.H2('Session Metrics', style={'textAlign': 'center'}),
                html.Div(id='metrics-display', style={
                    'backgroundColor': '#f4f4f4', 
                    'padding': '20px', 
                    'borderRadius': '10px'
                })
            ], style={'width': '100%', 'marginBottom': '20px'}),
            
            # Graphs Container
            html.Div([
                dcc.Graph(id='ear-mar-graph', style={'width': '50%', 'display': 'inline-block'}),
                dcc.Graph(id='status-distribution', style={'width': '50%', 'display': 'inline-block'}),
                dcc.Graph(id='yawn-intensity-graph', style={'width': '50%', 'display': 'inline-block'}),
                dcc.Graph(id='blink-trend-graph', style={'width': '50%', 'display': 'inline-block'})
            ])
        ])
    ])
    
    @app.callback(
        [Output('ear-mar-graph', 'figure'),
         Output('status-distribution', 'figure'),
         Output('yawn-intensity-graph', 'figure'),
         Output('blink-trend-graph', 'figure'),
         Output('metrics-display', 'children')],
        [Input('interval-component', 'n_intervals')]
    )
    def update_dashboard(n):
        try:
            if not os.path.exists(log_file_path):
                return [{} for _ in range(4)] + ["No data available"]
            
            df = pd.read_csv(log_file_path)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            
            # Calculate metrics
            metrics = calculate_metrics(df)
            
            # EAR and MAR Graph
            ear_mar_fig = go.Figure()
            ear_mar_fig.add_trace(go.Scatter(
                x=df['Timestamp'], y=df['Eye_Aspect_Ratio'], 
                mode='lines', name='Eye Aspect Ratio',
                line=dict(color='blue')
            ))
            ear_mar_fig.add_trace(go.Scatter(
                x=df['Timestamp'], y=df['Mouth_Aspect_Ratio'], 
                mode='lines', name='Mouth Aspect Ratio',
                line=dict(color='red')
            ))
            ear_mar_fig.update_layout(title='EAR and MAR Over Time')
            
            # Status Distribution Pie Chart
            status_fig = go.Figure(data=[go.Pie(
                labels=list(metrics['status_breakdown'].keys()),
                values=list(metrics['status_breakdown'].values()),
                hole=0.3
            )])
            status_fig.update_layout(title='Status Distribution')
            
            # Yawn Intensity Bar Graph
            yawn_fig = go.Figure(data=[go.Bar(
                x=list(metrics['yawn_intensity'].keys()),
                y=list(metrics['yawn_intensity'].values()),
                marker_color=['yellow', 'orange', 'red']
            )])
            yawn_fig.update_layout(title='Yawn Intensity Distribution')
            
            # Blink Trend Graph
            blink_fig = go.Figure(data=[go.Scatter(
                x=df['Timestamp'], y=df['Blink_Count'], 
                mode='lines+markers', name='Blinks'
            )])
            blink_fig.update_layout(title='Blink Count Over Time')
            
            # Metrics Display
            metrics_display = [
                html.Div([
                    html.H3(f"Overall Accuracy: {metrics['overall_accuracy']}%", 
                            style={'color': 'green' if metrics['overall_accuracy'] > 70 else 'red'}),
                    html.H4("Status Breakdown:"),
                    html.Ul([
                        html.Li(f"{status}: {percentage:.2f}%") 
                        for status, percentage in metrics['status_breakdown'].items()
                    ]),
                    html.H4("Temporal Analysis:"),
                    html.Ul([
                        html.Li(f"Total Duration: {metrics['temporal_analysis']['total_duration']:.2f} seconds"),
                        html.Li(f"Avg Blinks per Minute: {metrics['temporal_analysis']['avg_blinks_per_minute']:.2f}")
                    ])
                ])
            ]
            
            return ear_mar_fig, status_fig, yawn_fig, blink_fig, metrics_display
        
        except Exception as e:
            print(f"Dashboard update error: {e}")
            return [{} for _ in range(4)] + [f"Error: {str(e)}"]
    
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
