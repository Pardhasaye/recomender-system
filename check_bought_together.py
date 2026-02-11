import pandas as pd
from data_loader import load_jsonl

# Load metadata
df = load_jsonl('meta_Software.jsonl')

print("Checking all columns for 'also' or 'related' fields...")
print(f"\nAll columns: {df.columns.tolist()}\n")

# Check for any columns with 'also', 'related', 'together', 'similar'
keywords = ['also', 'related', 'together', 'similar', 'recommend']
for col in df.columns:
    col_lower = col.lower()
    if any(kw in col_lower for kw in keywords):
        non_null = df[col].notna().sum()
        print(f"Found: {col} - {non_null} non-null values")
        
        # Show a sample
        sample = df[col].dropna().head(1)
        if not sample.empty:
            print(f"  Sample: {sample.iloc[0]}\n")

# Also check 'details' field which might contain related info
if 'details' in df.columns:
    print("\nChecking 'details' field structure...")
    sample_details = df['details'].dropna().head(1)
    if not sample_details.empty:
        print(f"Sample details: {sample_details.iloc[0]}")
