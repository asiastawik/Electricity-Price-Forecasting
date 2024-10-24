# Electricity Price Forecasting Analysis

## Project Overview
This project involves analyzing electricity price data from the GEFCom and Nord Pool datasets (2013-2016). Using Python libraries like `pandas` and `numpy`, we will create naive and AR(1) forecasts, evaluate their performance, and visualize the results through various plots.

## Downloading the Dataset
- Download and unpack the datasets from the Electricity price data (GEFCom, Nord Pool 2013-2016) available in ePortal under T03+L03 The forecaster’s toolbox.

## Loading the Data
- Load the contents of the `GEFCOM.txt` file using either `numpy` or `pandas`. 
- Ensure to examine the columns as described on ePortal.

## Data Visualization
- **Plotting**: Create two subplots in the same figure:
  - **Zonal Price**: Plot the zonal price data.
  - **Zonal Load**: Plot the zonal load data.
- Format the x-ticks correctly, either by manually setting them (e.g., one tick every 6 months) or by parsing the dates. The x-tick labels should follow the format DD/MM YYYY (e.g., 12/01 2011).

## Naive Forecast Preparation
- Create a new column for a naive forecast that takes the value of the corresponding hourly period from the preceding day. Label this forecast as "naive".

## AR(1) Forecast Preparation
- Prepare two new columns with AR(1) forecasts defined as:
  \[
  \hat{y}_{d,h} = \beta_0 + \beta_1 \cdot y_{d-1,h} + \epsilon
  \]
- **Single Model Approach**:
  1. Estimate the weights using data from all hours up until (and including) 2011-06-30, resulting in 4320 samples.
  2. Produce forecasts for all days after 2011-06-30 and compute the Mean Absolute Error (MAE).
  
- **Separate Hours Approach**:
  1. Estimate a separate set of weights for each hour of the day (24 sets total).
  2. Produce forecasts for all days after 2011-06-30 and compute the MAE.

## Longer Calibration Window
- Repeat the AR(1) forecasting process from point 1.5, using a longer calibration window by taking all data prior to the end of the year 2012 for training. 
- Compare the relative performance of the single model approach against the separate hours approach.

## Rolling Calibration Window
- Prepare a separate hours forecast using a rolling calibration window, with the initial calibration window extending until the end of the year 2012.
