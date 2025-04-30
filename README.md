# Distro
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample data (initial estimates)
years = np.arange(1774, 1990)
wealth_shares = {
    '0-0%': np.linspace(5, 0, len(years)),  # Gradually decreasing to 0
    '0-50%': np.linspace(10, 5, len(years)),
    '50-90%': np.linspace(30, 35, len(years)),
    '90-99%': np.linspace(30, 30, len(years)),
    '99-99.9%': np.linspace(15, 15, len(years)),
    '99.9-100%': np.linspace(15, 15, len(years)),
}


# Create DataFrame
df = pd.DataFrame(wealth_shares, index=years)
df.index.name = 'Year'


# Set values to 0 after 1865 for '0-0%' class
df.loc[df.index > 1865, '0-0%'] = 0


# Apply linear regression to each income class
for col in df.columns:
    X = df.index.values.reshape(-1, 1)  # Reshape years for regression
    y = df[col].values
    model = LinearRegression()
    model.fit(X, y)
    df[col] = model.predict(X)  # Update with predicted values

# Create stacked area line chart
plt.figure(figsize=(10, 6))
plt.stackplot(
    df.index,
    [df[col] for col in df.columns],
    labels=df.columns,
    alpha=0.7
)
plt.xlabel('Year')
plt.ylabel('Wealth Share (%)')
plt.title('Estimated Wealth Distribution in the United States (1774-1989)')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()
