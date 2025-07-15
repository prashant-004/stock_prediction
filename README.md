# 📈 Stock Price Prediction using LSTM (Recursive Forecasting)

This project builds a robust time-series deep learning pipeline using **Long Short-Term Memory (LSTM)** networks to predict future stock prices. The model is designed to forecast **the next 5 business days** of stock prices based on **past 90 days** of historical and technical features.

---

## Project Objective

* Predict short-term **stock closing prices** using historical stock data.
* Leverage **LSTM networks** for their ability to learn temporal dependencies.
* Implement a **recursive prediction strategy** to simulate autoregressive forecasting.
* Optimize model performance with **hyperparameter tuning** and **early stopping**.

---

## Data & Features

* **Raw Inputs:**
  Historical stock data including: `Open`, `High`, `Low`, `Close`, `Volume`, and `Date`.

* **Engineered Technical Indicators:**

  * `EMA (10, 20)` – Exponential Moving Averages
  * `RSI` – Relative Strength Index
  * `MACD` and `Signal Line`
  * `Bollinger Bands (Upper/Lower)`
  * `Return` – Daily % change in close
  * `Volume_log` – Log-transformed volume
  * `Return_clipped` – Returns clipped between \[-5%, +5%]

* **Target Variable:**

  * `Target` price of the **next day** (for training)
  * 5-day recursive forecast for evaluation

---

##  Model Architecture

* **Model Type:** LSTM with recursive single-step prediction
* **Architecture (Tuned with KerasTuner):**

  * 1–5 LSTM layers (variable)
  * Dropout regularization (0.1–0.5)
  * Final Dense layer with **1 neuron** (predicts 1-day ahead)
* **Optimizer:** Adam (learning rate tuned: 0.1, 0.01, 0.001)
* **Loss Function:** Mean Squared Error (MSE)

---

## Recursive Forecasting Strategy

The model predicts **1 day at a time**. For each prediction:

1. The latest 90-day sequence is passed to the LSTM model.
2. The **predicted close price** is appended back into the sequence.
3. This process is repeated **5 times** to simulate a 5-day forecast.

---

##  Model Training & Evaluation

* **Train-Test Split:** 80:20 (chronological, no shuffle)
* **Epochs:** Up to 50 (early stopping after 5 stale epochs)
* **Batch Size:** 32
* **Validation Split:** 20%


## Future Improvements

* Switch to **Sequence-to-Sequence (Seq2Seq)** model with **Attention** for better multi-day prediction.
* Introduce **multi-feature recursive prediction** (predict other variables like volume or technical indicators).
* Experiment with **transformer models** or **temporal convolutional networks** (TCN).


