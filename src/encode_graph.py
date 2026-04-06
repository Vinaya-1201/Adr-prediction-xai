import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load cleaned dataset
data = pd.read_csv("data/drug_adr_cleaned.csv")

# Encode drugs
drug_encoder = LabelEncoder()
data['drug_id'] = drug_encoder.fit_transform(data['drug_name'])

# Encode ADRs
adr_encoder = LabelEncoder()
data['adr_id'] = adr_encoder.fit_transform(data['side_effects'])

print("Total unique drugs:", len(data['drug_id'].unique()))
print("Total unique ADRs:", len(data['adr_id'].unique()))

print("\nSample encoded data:")
print(data.head())

# Save encoded dataset
data.to_csv("data/drug_adr_encoded.csv", index=False)

print("\nSaved as drug_adr_encoded.csv")