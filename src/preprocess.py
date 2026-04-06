import pandas as pd

# Load dataset
data = pd.read_csv("data/sider.csv")

# Keep only required columns
data = data[['drug_name', 'side_effects']]

# Drop rows with missing side effects
data = data.dropna(subset=['side_effects'])

# Convert side effects into list (split by comma)
data['side_effects'] = data['side_effects'].str.lower()
data['side_effects'] = data['side_effects'].str.split(',')

# Explode → one row per side effect
data = data.explode('side_effects')

# Clean spaces
data['side_effects'] = data['side_effects'].str.strip()

# Remove empty values
data = data[data['side_effects'] != '']

print("Cleaned Dataset Shape:", data.shape)
print(data.head())

# Save cleaned file
data.to_csv("data/drug_adr_cleaned.csv", index=False)

print("\nSaved as drug_adr_cleaned.csv")
import re

def clean_side_effects(text):
    if pd.isna(text):
        return None
    
    # Take only first sentence
    text = text.split(".")[0]
    
    # Remove special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    
    # Lowercase
    text = text.lower().strip()
    
    return text

data['side_effects'] = data['side_effects'].apply(clean_side_effects)