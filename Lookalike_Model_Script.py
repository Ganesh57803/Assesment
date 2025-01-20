import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Load the datasets
customers = pd.read_csv('Customers.csv')
transactions = pd.read_csv('Transactions.csv')

# Merge datasets to create a customer profile
customer_transactions = transactions.merge(customers, on='CustomerID')

# Create a feature set for similarity calculation
customer_features = customer_transactions.groupby('CustomerID').agg({
    'TransactionID': 'count',
    'TotalValue': 'mean',
    'Region': 'first'
}).reset_index()
customer_features.rename(columns={'TransactionID': 'TotalTransactions', 'TotalValue': 'AverageTransactionValue'}, inplace=True)

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(customer_features[['TotalTransactions', 'AverageTransactionValue']])

# Calculate cosine similarity
similarity_matrix = cosine_similarity(scaled_features)

# Create a DataFrame for similarity scores
similarity_df = pd.DataFrame(similarity_matrix, index=customer_features['CustomerID'], columns=customer_features['CustomerID'])

# Function to get lookalikes for a given customer
def get_lookalikes(customer_id, top_n=3):
    similar_customers = similarity_df[customer_id].nlargest(top_n + 1).iloc[1:]
    return similar_customers.index.tolist(), similar_customers.values.tolist()

# Get lookalikes for the first 20 customers
lookalike_results = {}
for customer_id in customer_features['CustomerID'][:20]:
    lookalikes, scores = get_lookalikes(customer_id)
    lookalike_results[customer_id] = list(zip(lookalikes, scores))

# Save the results to Lookalike.csv
lookalike_df = pd.DataFrame.from_dict(lookalike_results, orient='index')
lookalike_df.to_csv('Lookalike.csv', header=False)
print(lookalike_df.head())
