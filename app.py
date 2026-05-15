import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.title("Vaca Muerta Forecast and Underperforming deposit detection")

df = pd.read_csv("data/company_production.csv")

df_forecast_arps = pd.read_csv("data/forecast_arps.csv")
df_forecast_ph = pd.read_csv("data/forecast_prophet.csv")
df_underperformance = pd.read_csv("data/underperformance.csv")

def plot_forecast_arps(df_forecast):
    for deposit in df_forecast['deposit'].unique():
        d_dep = df_forecast_arps[df_forecast_arps['deposit']  == deposit]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d_dep['ds'], y=d_dep['actual'], name='Actual', mode='markers'))
        fig.add_trace(go.Scatter(x=d_dep['ds'], y=d_dep['yhat'], name='Hyperbolic fit'))
        fig.update_layout(title=f"{deposit}", 
                        yaxis_title="Production (m³)", xaxis_title="Date")
        st.plotly_chart(fig, width='stretch')

def plot_forecast_prophet(df_forecast):
    for deposit in df_forecast['deposit'].unique():
        d_dep = df_forecast[df_forecast['deposit'] == deposit]

        TRAIN_CUTOFF = '2024-12-31' # Defined in the notebook and script

        hist_forecast   = d_dep[d_dep['ds'] <= TRAIN_CUTOFF]
        future_forecast = d_dep[d_dep['ds'] >  TRAIN_CUTOFF]

        fig = go.Figure()
    
        fig.add_trace(go.Scatter(
            x=d_dep['ds'], y=d_dep['actual'],
            mode='markers', name='Actual',
            marker=dict(color='steelblue', size=6)
        ))
        fig.add_trace(go.Scatter(
            x=pd.concat([future_forecast['ds'], future_forecast['ds'].iloc[::-1]]),
            y=pd.concat([future_forecast['yhat_upper'], future_forecast['yhat_lower'].iloc[::-1]]),
            fill='toself', fillcolor='rgba(255,100,100,0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence interval'
        ))
        fig.add_trace(go.Scatter(
            x=hist_forecast['ds'], y=hist_forecast['yhat'],
            mode='lines', name='Prophet fit',
            line=dict(color='red')
        ))
        fig.add_trace(go.Scatter(
            x=future_forecast['ds'], y=future_forecast['yhat'],
            mode='lines', name='Prophet forecast',
            line=dict(color='red', dash='dash')
        ))
        fig.add_vline(
            x=pd.Timestamp(TRAIN_CUTOFF).timestamp() * 1000,  # convert to milliseconds
            line_dash='dot', line_color='gray',
            annotation_text='Train cutoff', 
            annotation_position='top right'
        )
        fig.update_layout(
            title=f"{deposit}",
            yaxis_title="Production (m³)",
            xaxis_title="Date"
        )
        st.plotly_chart(fig, width='stretch')

def plot_underperformance_detection(df_underperformance):
    # Underperformance Detection
    PROD_THRESHOLD = 0.80      # 20% below forecast
    CONSEC_MONTHS  = 3         # 3 consecutive months
    WATER_CUT_WINDOW = 3       # months to detect rising water cut

    results_flags = []

    for deposit in df_underperformance['deposit'].unique():
        df_eval = df_underperformance[df_underperformance['deposit'] == deposit]
        
        # Signal 1: production below threshold
        df_eval['below_threshold'] = df_eval['prod_pet'] < df_eval['prod_forecast'] * PROD_THRESHOLD
        
        # Signal 2: rising water cut (rolling slope > 0 for N months)
        df_eval['water_cut_rising'] = (
            df_eval['water_cut']
            .rolling(WATER_CUT_WINDOW)
            .apply(lambda x: x.iloc[-1] > x.iloc[0])  # rising if last > first in window
            .fillna(False)
            .astype(bool)
        )
        
        # Consecutive months below threshold
        df_eval['consec_below'] = (
            df_eval['below_threshold']
            .rolling(CONSEC_MONTHS)
            .sum()
            .fillna(0) == CONSEC_MONTHS
        )
        
        # Final flag: both signals active
        df_eval['underperforming'] = df_eval['consec_below'] & df_eval['water_cut_rising']
        
        flagged_months = df_eval[df_eval['underperforming']]['date'].tolist()
        results_flags.append({
            'deposit': deposit,
            'flagged_months': len(flagged_months),
            'first_flag': flagged_months[0] if flagged_months else None,
            'last_flag':  flagged_months[-1] if flagged_months else None
        })
        
        # Plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df_eval['date'], y=df_eval['prod_pet'],
            mode='markers+lines', name='Actual',
            marker=dict(color='steelblue')
        ))
        fig.add_trace(go.Scatter(
            x=df_eval['date'], y=df_eval['prod_forecast'],
            mode='lines', name='Forecast',
            line=dict(color='red', dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=df_eval['date'], y=df_eval['prod_forecast'] * PROD_THRESHOLD,
            mode='lines', name='Underperformance threshold (80%)',
            line=dict(color='orange', dash='dot')
        ))
        
        # Highlight flagged months
        flagged = df_eval[df_eval['underperforming']]
        fig.add_trace(go.Scatter(
            x=flagged['date'], y=flagged['prod_pet'],
            mode='markers', name='Flagged underperforming',
            marker=dict(color='red', size=10, symbol='x')
        ))
        
        fig.update_layout(
            title=f"{deposit} — Underperformance Detection",
            yaxis_title="Production (m³)",
            xaxis_title="Date"
        )
        st.plotly_chart(fig, width='stretch')
        
    

st.subheader('Arp declining curves')
plot_forecast_arps(df_forecast_arps)


st.subheader('Prophet forecast')
plot_forecast_prophet(df_forecast_ph)

st.subheader('Underperforming well detection')
plot_underperformance_detection(df_underperformance)