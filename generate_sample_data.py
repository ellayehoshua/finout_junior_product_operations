"""
Sample Data Generator for OpenAI Cost Analysis
Creates a realistic sample dataset for testing the analysis tool.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
START_DATE = datetime(2025, 6, 1)
END_DATE = datetime(2025, 6, 30)
N_PROJECTS = 8
N_USERS_PER_PROJECT = 3
N_MODELS = 5

# Generate date range
dates = pd.date_range(START_DATE, END_DATE, freq='D')

# Project and user IDs
projects = [f"project_{chr(65+i)}" for i in range(N_PROJECTS)]
models = ['gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo', 'o1-preview', 'o1-mini']
service_tiers = ['default', 'scale']

# Cost per 1k tokens for each model (approximation)
model_costs = {
    'gpt-4o': {'input': 0.0050, 'output': 0.0150},
    'gpt-4o-mini': {'input': 0.00015, 'output': 0.00060},
    'gpt-3.5-turbo': {'input': 0.0005, 'output': 0.0015},
    'o1-preview': {'input': 0.0150, 'output': 0.0600},
    'o1-mini': {'input': 0.0030, 'output': 0.0120}
}

# Premium models
premium_models = ['gpt-4o', 'o1-preview', 'o1-mini']

# Generate data
data = []

for date in dates:
    day_of_month = date.day
    
    # Simulate weekly pattern and growth
    weekly_factor = 1.0 + 0.3 * np.sin(date.weekday() * np.pi / 7)  # Weekday variation
    growth_factor = 1.0 + (day_of_month / 30) * 0.2  # 20% growth over month
    
    # Simulate spike on day 15
    if day_of_month == 15:
        weekly_factor *= 1.5
    
    # Anomaly spike on day 23
    if day_of_month == 23:
        weekly_factor *= 1.8
    
    for project_idx, project in enumerate(projects):
        # Each project has different activity levels
        project_activity = np.random.exponential(scale=1.5) * (10 - project_idx) / 10
        
        n_daily_requests = int(
            np.random.poisson(100 * project_activity * weekly_factor * growth_factor)
        )
        
        if n_daily_requests == 0:
            continue
        
        # Distribute requests across users
        for _ in range(n_daily_requests):
            user_id = f"user_{project}_{np.random.randint(1, N_USERS_PER_PROJECT+1)}"
            api_key_id = f"sk-{project}-{np.random.randint(1, 4)}"
            
            # Model selection (premium models less frequent)
            model_probs = [0.25, 0.35, 0.25, 0.05, 0.10]
            model = np.random.choice(models, p=model_probs)
            
            service_tier = np.random.choice(service_tiers, p=[0.7, 0.3])
            
            # Token generation
            input_tokens = int(np.random.lognormal(mean=7, sigma=1.2))  # ~1000 avg
            
            # I/O ratio varies by model
            if model.startswith('o1'):
                output_ratio = np.random.uniform(0.3, 0.8)  # Reasoning models
            else:
                output_ratio = np.random.uniform(0.5, 2.0)  # Regular models
            
            output_tokens = int(input_tokens * output_ratio)
            
            # Cache simulation (higher for some projects)
            cache_rate = 0.3 if project_idx < 3 else 0.1
            cache_rate += np.random.uniform(-0.1, 0.1)
            cache_rate = max(0, min(1, cache_rate))
            
            input_cached_tokens = int(input_tokens * cache_rate)
            input_uncached_tokens = input_tokens - input_cached_tokens
            
            # Calculate cost
            input_cost = (input_tokens / 1000) * model_costs[model]['input']
            output_cost = (output_tokens / 1000) * model_costs[model]['output']
            
            # Cache discount (50% off for cached tokens)
            cache_discount = (input_cached_tokens / 1000) * model_costs[model]['input'] * 0.5
            
            cost_usd = input_cost + output_cost - cache_discount
            
            # Add some noise
            cost_usd *= np.random.uniform(0.95, 1.05)
            
            # Premium flag
            is_premium_model = model in premium_models
            
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'project_id': project,
                'user_id': user_id,
                'api_key_id': api_key_id,
                'model': model,
                'service_tier': service_tier,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'input_cached_tokens': input_cached_tokens,
                'input_uncached_tokens': input_uncached_tokens,
                'num_model_requests': 1,
                'cost_usd': round(cost_usd, 6),
                'is_premium_model': is_premium_model
            })

# Create DataFrame
df = pd.DataFrame(data)

print(f"Generated {len(df):,} rows of sample data")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Total cost: ${df['cost_usd'].sum():,.2f}")
print(f"Projects: {df['project_id'].nunique()}")
print(f"Users: {df['user_id'].nunique()}")
print(f"Models: {', '.join(df['model'].unique())}")

# Save to Excel
output_file = "OpenAI Cost and Usage Data - June 2025.xlsx"
df.to_excel(output_file, sheet_name='Usage Data', index=False)

print(f"\n✓ Saved sample data to: {output_file}")
print(f"\nYou can now run: python main.py")

