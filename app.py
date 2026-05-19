import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

COLORS = {
    'Oil':   '#D2691E',
    'Gas':   '#E63946',
    'Water': '#1E90FF'
}

MARKER_SIZE = 6

INITIAL_DEPOSIT = "ESTACION FERNANDEZ ORO"

st.title("Vaca Muerta Forecast and Underperforming deposit detection")


df_basin_production = pd.read_csv("data/basin_monthly.csv")
df_deposit_locations = pd.read_csv("data/deposit_locations.csv")

# Load forecasting data precomputed
df_arps = pd.read_csv("data/forecast_arps.csv")
df_forecast_ph = pd.read_csv("data/forecast_prophet.csv")
df_underperformance = pd.read_csv("data/underperformance.csv")

# Load metrics
df_mape_arps = pd.read_csv("data/arps_mapes.csv")
df_mape_ph = pd.read_csv("data/prophet_mapes.csv")


def plot_deposit_production(df_production, deposit):
    d_dep = df_production[df_production['areayacimiento'] == deposit]

    # Filtering zero values for plotting purposes
    THRESHOLD = 0.01
    oil_prod = d_dep[d_dep['prod_pet'] > THRESHOLD]
    gas_prod = d_dep[d_dep['prod_gas'] > THRESHOLD]
    water_prod = d_dep[d_dep['prod_agua'] > THRESHOLD]

    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=oil_prod['date'], y=oil_prod['prod_pet'], name = 'Oil', mode='markers', marker=dict(color=COLORS['Oil'], size=MARKER_SIZE)))
    fig.add_trace(go.Scatter(x=gas_prod['date'], y=gas_prod['prod_gas'], name = 'Gas', mode='markers', marker=dict(color=COLORS['Gas'], size=MARKER_SIZE)))
    fig.add_trace(go.Scatter(x=water_prod['date'], y=water_prod['prod_agua'], name = 'Water', mode='markers', marker=dict(color=COLORS['Water'], size=MARKER_SIZE)))

    fig.update_layout(title=f"{deposit} - Production History", yaxis_title = "Production (m³)", xaxis_title = "Date")
    st.plotly_chart(fig, width='stretch')


     

def plot_arps(df_arps, df_mape, deposit):
    if deposit in df_arps['deposit'].unique():
        d_dep = df_arps[df_arps['deposit']  == deposit]
        mape_value =  df_mape[df_mape['deposit'] == deposit]['mape'].values[0]
        
        st.metric("Arps MAPE",f"{mape_value}")
        if mape_value > 50:
            st.caption("High MAPE — Arps works best for deposits in clear decline phase")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d_dep['ds'], y=d_dep['actual'], name='Actual', mode='markers'))
        fig.add_trace(go.Scatter(x=d_dep['ds'], y=d_dep['yhat'], name='Hyperbolic fit'))
        fig.update_layout(title=f"{deposit} - Arps Decline Curve", 
                        yaxis_title="Production (m³)", xaxis_title="Date")
        st.plotly_chart(fig, width='stretch')
    else:
        st.caption("Deposit not mature enough")

def plot_forecast_prophet(df_forecast, df_mape, deposit):
    if deposit in df_forecast['deposit'].unique():
        d_dep = df_forecast[df_forecast['deposit'] == deposit]
        mape_value =  df_mape[df_mape['deposit'] == deposit]['mape'].values[0]

        st.metric("Prophet MAPE",f"{mape_value}")
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
            title=f"{deposit} - Prophet Forecast",
            yaxis_title="Production (m³)",
            xaxis_title="Date"
        )
        st.plotly_chart(fig, width='stretch')

def plot_underperformance_detection(df_underperformance, deposit):
    # Underperformance Detection
    PROD_THRESHOLD = 0.80      # 20% below forecast
    CONSEC_MONTHS  = 3         # 3 consecutive months
    WATER_CUT_WINDOW = 3       # months to detect rising water cut

    results_flags = []

    if deposit in df_underperformance['deposit'].unique():
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

if 'selected_deposit' not in st.session_state:
    st.session_state.selected_deposit = INITIAL_DEPOSIT

def plot_deposit_locations(df_locations):

    totals = df_basin_production.groupby('areayacimiento').agg(
        total_pet=('prod_pet', 'sum'),
        total_gas=('prod_gas', 'sum'),
        total_agua=('prod_agua', 'sum')
    ).reset_index()

    totals.columns = ['deposit', 'total_pet', 'total_gas', 'total_agua']

    # dominant resource
    totals['dominant'] = totals[['total_pet', 'total_gas', 'total_agua']].idxmax(axis=1).map({
        'total_pet': 'Oil',
        'total_gas': 'Gas',
        'total_agua': 'Water'
    })

    # total production
    totals['total_prod'] = totals['total_pet'] + totals['total_gas'] + totals['total_agua']

    # Normalization because of scale differences
    totals['size'] = np.log1p(totals['total_prod'])
    totals['size'] = (totals['size'] - totals['size'].min()) / (totals['size'].max() - totals['size'].min()) * 20 + 5  # range 5-25


    df_locations = df_locations.merge(totals, on='deposit', how='left')

    
    # Either select on the map or search by name
    deposits = df_locations['deposit'].tolist()

    fig = px.scatter_map(
        df_locations,
        lat='lat',
        lon='lon',
        hover_name='deposit',
        size='size',
        size_max=20,
        color='dominant',
        color_discrete_map=COLORS,
        hover_data={'lat': False, 'lon': False, 'size': False},
        zoom=8,
        map_style='carto-darkmatter'
    )

    event = st.plotly_chart(fig, on_select='rerun', width='stretch')

    # Detect selection
    if event.selection.points:
        st.session_state.selected_deposit = event.selection.points[0]['hovertext']

    search = st.selectbox(
        'Search by name',
        options=deposits,
        index=deposits.index(st.session_state.selected_deposit)
    )
    st.session_state.selected_deposit = search

st.text('Select a deposit to view their historical production, bubble sized related to production output')
plot_deposit_locations(df_deposit_locations)
plot_deposit_production(df_basin_production, st.session_state.selected_deposit)

deposit = st.selectbox('Select a deposit to showcase prediction models', df_underperformance['deposit'].unique() ,index=df_underperformance['deposit'].unique().tolist().index(INITIAL_DEPOSIT))


st.subheader('Arp Decline Curve')
plot_arps(df_arps, df_mape_arps, deposit)


st.subheader('Prophet forecast')
plot_forecast_prophet(df_forecast_ph, df_mape_ph, deposit)

st.subheader('Underperforming well detection')
plot_underperformance_detection(df_underperformance, deposit)