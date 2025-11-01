"""
Test script to demonstrate the enhanced load_data() function
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create a sample Excel file with various column name formats
print("Creating sample test data...")

# Generate sample data with mixed column name formats
dates = pd.date_range('2025-06-01', '2025-06-10', freq='D')

data = {
    'Date': dates,  # Will become 'date'
    'ProjectID': ['proj_A', 'proj_B', 'proj_C'] * (len(dates) // 3 + 1),  # Will become 'project_id'
    'User ID': ['user_1', 'user_2', 'user_3'] * (len(dates) // 3 + 1),  # Will become 'user_id'
    'API-Key-ID': ['key_1', 'key_2', 'key_3'] * (len(dates) // 3 + 1),  # Will become 'api_key_id'
    'Model': ['gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo'] * (len(dates) // 3 + 1),
    'Service Tier': ['default', 'scale', 'default'] * (len(dates) // 3 + 1),  # Will become 'service_tier'
    'InputTokens': np.random.randint(100, 2000, len(dates)),  # Will become 'input_tokens'
    'OutputTokens': np.random.randint(50, 1000, len(dates)),  # Will become 'output_tokens'
    'input cached tokens': np.random.randint(0, 500, len(dates)),  # Will become 'input_cached_tokens'
    'input_uncached_tokens': np.random.randint(100, 1500, len(dates)),
    'Num-Model-Requests': np.random.randint(1, 10, len(dates)),  # Will become 'num_model_requests'
    'CostUSD': np.random.uniform(0.01, 5.0, len(dates)),  # Will become 'cost_usd'
    'is premium model': [True, False, False] * (len(dates) // 3 + 1),  # Will become 'is_premium_model'
    'Extra Column 1': ['value'] * len(dates),  # Extra column not in expected schema
    'Extra Column 2': [123] * len(dates),  # Another extra column
}

# Trim to exact length
for key in data:
    if isinstance(data[key], list):
        data[key] = data[key][:len(dates)]

df = pd.DataFrame(data)

# Save to Excel
test_file = "test_data_sample.xlsx"
df.to_excel(test_file, sheet_name='Usage Data', index=False)

print(f"✓ Created test file: {test_file}")
print(f"  Original columns: {list(df.columns)}")
print()

# Now test the load_data function
print("="*80)
print("TESTING ENHANCED load_data() FUNCTION")
print("="*80)

from main import load_data

# Load the data
loaded_df, schema_report = load_data(test_file)

# Display results
if loaded_df is not None:
    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)
    
    print("\n1. COLUMN NAME TRANSFORMATION:")
    print("-" * 40)
    for orig, snake in schema_report['column_mapping'].items():
        marker = "✓" if orig == snake else "→"
        print(f"  {marker} {orig:30} => {snake}")
    
    print("\n2. DATE COLUMN:")
    print("-" * 40)
    print(f"  Date column found: {schema_report['date_column']}")
    print(f"  'day' column created: {'day' in loaded_df.columns}")
    if 'day' in loaded_df.columns:
        print(f"  Sample 'day' values: {loaded_df['day'].head(3).tolist()}")
    
    print("\n3. SCHEMA REPORT SUMMARY:")
    print("-" * 40)
    print(f"  Found columns: {len(schema_report['found_columns'])}/{13}")
    print(f"  Missing columns: {len(schema_report['missing_columns'])}")
    print(f"  Extra columns: {len(schema_report['extra_columns'])}")
    
    if schema_report['missing_columns']:
        print(f"\n  Missing: {', '.join(schema_report['missing_columns'])}")
    
    if schema_report['extra_columns']:
        print(f"\n  Extra: {', '.join(schema_report['extra_columns'])}")
    
    print("\n4. DATAFRAME INFO:")
    print("-" * 40)
    print(f"  Shape: {loaded_df.shape[0]} rows × {loaded_df.shape[1]} columns")
    print(f"  Columns: {list(loaded_df.columns)}")
    
    print("\n5. DATE RANGE:")
    print("-" * 40)
    if schema_report['date_range']:
        print(f"  Min date: {schema_report['date_range']['min']}")
        print(f"  Max date: {schema_report['date_range']['max']}")
    
    print("\n" + "="*80)
    print("✓ TEST PASSED - Enhanced load_data() working correctly!")
    print("="*80)
else:
    print("\n❌ TEST FAILED - Could not load data")

# Cleanup
import os
if os.path.exists(test_file):
    os.remove(test_file)
    print(f"\n✓ Cleaned up test file: {test_file}")

