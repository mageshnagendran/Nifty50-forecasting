import pandas as pd
from prophet import Prophet # type: ignore
import matplotlib.pyplot as plt
from prophet.plot import add_changepoints_to_plot
import os

# Loading data
data_path = "/Users/maki/python-maki/mynewbook/nifty.csv" # CAHNGE PATH ENVIRONMENT
df = pd.read_csv(data_path)

# Rename columns for Prophet
df = df.rename(columns={'Date': 'ds', 'Close': 'y'})
df['ds'] = pd.to_datetime(df['ds'])

print(df)

# Train Prophet model
model = Prophet()
model.fit(df)

# Create future dataframe (next 10 years)
future = model.make_future_dataframe(periods=365 * 10)
forecast = model.predict(future)

# Creating output directory
output_dir = "outputs/nifty"
os.makedirs(output_dir, exist_ok=True)

# Save forecast
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(f"{output_dir}/prophet_forecast.csv", index=False)

# Plot and save forecast
fig1 = model.plot(forecast, include_legend=True)
plt.title("Prophet Forecast")
plt.savefig(f"{output_dir}/prophet_forecast.png")

# Plot and save components
fig2 = model.plot_components(forecast)
plt.savefig(f"{output_dir}/prophet_components.png")

fig3 = model.plot(forecast)
a = add_changepoints_to_plot(fig3.gca(), model, forecast)
plt.savefig(f"{output_dir}/changepoints.png")

print("✅ Prophet model completed. Forecasts and plots saved in 'outputs'.")
