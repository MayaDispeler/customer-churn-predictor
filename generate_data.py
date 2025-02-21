# generate_data.py
import numpy as np
import pandas as pd

# For reproducibility
np.random.seed(42)

# Number of samples
num_samples = 1000

# Generate features
Usage = np.random.uniform(0, 100, size=num_samples)             # 0 to 100 hours
SupportTickets = np.random.poisson(lam=2, size=num_samples)     # average ~2 tickets
NPS = np.random.randint(0, 11, size=num_samples)                # 0 to 10
AccountAge = np.random.randint(1, 37, size=num_samples)         # 1 to 36 months

# Logistic-like transformation for churn probability
score = (
    0.05 * (100 - Usage)      # lower usage => higher churn
    + 0.3 * SupportTickets    # more tickets => higher churn
    + 1.0 * (10 - NPS)        # lower NPS => higher churn
    + 0.05 * (36 - AccountAge) # younger accounts => higher churn
)

# Convert to probability
churn_prob = 1 / (1 + np.exp(-0.1 * (score - 10)))

# Randomly assign churn based on churn_prob
Churned = (np.random.rand(num_samples) < churn_prob).astype(int)

# Create DataFrame
df = pd.DataFrame({
    'Usage': Usage,
    'SupportTickets': SupportTickets,
    'NPS': NPS,
    'AccountAge': AccountAge,
    'Churned': Churned
})

# Save to CSV
df.to_csv("data/customer_churn_data.csv", index=False)
print("Synthetic churn dataset created: data/customer_churn_data.csv")
