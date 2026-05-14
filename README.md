# Vaca Muerta — Oil & Gas Production Forecast

**Data:** Argentina's Secretary of Energy (datos.energia.gob.ar)  
**Period:** 2006–2026  
**Basin:** Cuenca Neuquina  

## What this project does
Analyzes unconventional oil and gas production across deposits in the 
Neuquén basin, and forecasts future production using decline curve analysis 
and machine learning.

## Current state
- Exploratory analysis: well locations, production by company, 
  deposit comparisons, water cut trends
- Arps decline curve fitting (hyperbolic + exponential) 
  for top producing deposits
- 12-month production forecast for top perfoming deposits

## Roadmap
- [ ] Prophet time series forecasting
- [ ] Baseline vs ML comparison (Arps vs Prophet)
- [ ] Evaluation metrics (RMSE, MAE)

## Data source
[datos.energia.gob.ar](http://datos.energia.gob.ar)
