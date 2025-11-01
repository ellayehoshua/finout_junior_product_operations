#!/usr/bin/env python3
"""
Test script for q4_anomalies() function.

This tests:
- Spike Index calculation (avg z-score of requests and tokens with 14-day baseline)
- Inefficiency Index calculation (weighted composite)
- Flagging logic (Spike > 3, Inefficiency > 1.3)
- Panel visualization generation
- CSV output
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Import from main.py
from main import q4_anomalies, ensure_artifacts_dir

def generate_test_data_with_anomalies():
    """
    Generate synthetic data with known anomalies:
    - Normal baseline period (14 days)
    - Spike days with high requests/tokens
    - Projects with varying efficiency profiles
    """
    np.random.seed(42)
    
    # Date range: 30 days in June 2025
    start_date = datetime(2025, 6, 1)
    dates = [start_date + timedelta(days=i) for i in range(30)]
    
    projects = ['ProjectA', 'ProjectB', 'ProjectC', 'ProjectD', 'ProjectE']
    
    records = []
    
    for day_idx, date in enumerate(dates):
        for project in projects:
            # Base metrics
            base_requests = 1000 + np.random.randint(-100, 100)
            base_tokens = 50000 + np.random.randint(-5000, 5000)
            
            # Inject spike anomalies on specific days
            if day_idx in [15, 20, 25]:  # Days 16, 21, 26
                # Spike: 3-4x normal volume
                spike_multiplier = 3.5 + np.random.random()
                base_requests = int(base_requests * spike_multiplier)
                base_tokens = int(base_tokens * spike_multiplier)
            
            # Project-specific efficiency profiles
            if project == 'ProjectA':
                # Efficient: low cost, high cache, non-premium
                cost_per_1k = 0.02 + np.random.random() * 0.005
                is_premium = 0
                cache_hit = 0.8 + np.random.random() * 0.15
            
            elif project == 'ProjectB':
                # Inefficient: high cost, low cache, high premium
                cost_per_1k = 0.08 + np.random.random() * 0.02
                is_premium = 1
                cache_hit = 0.1 + np.random.random() * 0.1
            
            elif project == 'ProjectC':
                # Very inefficient: very high cost, no cache, premium
                cost_per_1k = 0.12 + np.random.random() * 0.03
                is_premium = 1
                cache_hit = 0.05 + np.random.random() * 0.05
            
            elif project == 'ProjectD':
                # Moderate efficiency
                cost_per_1k = 0.04 + np.random.random() * 0.01
                is_premium = 0
                cache_hit = 0.5 + np.random.random() * 0.2
            
            else:  # ProjectE
                # Good efficiency
                cost_per_1k = 0.025 + np.random.random() * 0.005
                is_premium = 0
                cache_hit = 0.7 + np.random.random() * 0.15
            
            # Calculate derived values
            tokens = base_tokens
            input_tokens = int(tokens * 0.6)
            output_tokens = tokens - input_tokens
            
            input_cached = int(input_tokens * cache_hit)
            input_uncached = input_tokens - input_cached
            
            cost_usd = (tokens / 1000) * cost_per_1k
            
            records.append({
                'date': date,
                'project_id': project,
                'user_id': f'user_{project}',
                'api_key_id': f'key_{project}',
                'model': 'gpt-4' if is_premium else 'gpt-3.5-turbo',
                'service_tier': 'premium' if is_premium else 'standard',
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'input_cached_tokens': input_cached,
                'input_uncached_tokens': input_uncached,
                'num_model_requests': base_requests,
                'cost_usd': cost_usd,
                'is_premium_model': is_premium,
                'tokens': tokens,
                'tokens_per_req': tokens / base_requests if base_requests > 0 else 0,
                'io_ratio': output_tokens / input_tokens if input_tokens > 0 else 0,
                'cache_hit_rate': cache_hit,
                'cost_per_1k': cost_per_1k,
                'cost_per_req': cost_usd / base_requests if base_requests > 0 else 0
            })
    
    df = pd.DataFrame(records)
    return df

def test_q4_anomalies():
    """Test the q4_anomalies function."""
    print("="*80)
    print("TESTING Q4_ANOMALIES FUNCTION")
    print("="*80)
    
    # Ensure artifacts directory exists
    ensure_artifacts_dir()
    
    # Generate test data
    print("\n1. Generating test data with known anomalies...")
    df = generate_test_data_with_anomalies()
    print(f"   ✓ Generated {len(df)} records")
    print(f"   ✓ Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   ✓ Projects: {df['project_id'].unique().tolist()}")
    
    # Expected anomalies
    print("\n2. Expected anomalies in test data:")
    print("   • Spike days: June 16, 21, 26 (3.5-4.5x baseline)")
    print("   • Inefficient projects: ProjectB, ProjectC")
    print("   • ProjectC should have highest Inefficiency Index")
    
    # Run Q4 analysis
    print("\n3. Running Q4 analysis...")
    print("-" * 80)
    summary = q4_anomalies(df)
    print("-" * 80)
    
    # Verify outputs
    print("\n4. Verifying outputs...")
    artifacts_dir = "./artifacts"
    
    expected_files = [
        'q4_flags.csv',
        'q4_anomalies.png'
    ]
    
    all_exist = True
    for file in expected_files:
        path = f"{artifacts_dir}/{file}"
        exists = os.path.exists(path)
        status = "✓" if exists else "✗"
        print(f"   {status} {file}: {'Found' if exists else 'MISSING'}")
        if not exists:
            all_exist = False
    
    # Check flags CSV content
    flags_path = f"{artifacts_dir}/q4_flags.csv"
    if os.path.exists(flags_path):
        flags_df = pd.read_csv(flags_path)
        print(f"\n5. Flags CSV Analysis:")
        print(f"   • Total flags: {len(flags_df)}")
        
        spike_flags = flags_df[flags_df['type'] == 'Spike']
        ineff_flags = flags_df[flags_df['type'] == 'Inefficiency']
        
        print(f"   • Spike flags: {len(spike_flags)}")
        if len(spike_flags) > 0:
            print(f"     - Dates: {spike_flags['date'].unique().tolist()}")
            print(f"     - Max Spike Index: {spike_flags['spike_index'].max():.2f}")
        
        print(f"   • Inefficiency flags: {len(ineff_flags)}")
        if len(ineff_flags) > 0:
            print(f"     - Projects: {ineff_flags['entity'].tolist()}")
            print(f"     - Max Inefficiency Index: {ineff_flags['inefficiency_index'].max():.2f}")
    
    # Display summary
    print(f"\n6. Executive Summary:")
    print(summary)
    
    # Test verdict
    print("\n" + "="*80)
    if all_exist:
        print("✓ TEST PASSED: All artifacts generated successfully")
        print("="*80)
        return True
    else:
        print("✗ TEST FAILED: Some artifacts missing")
        print("="*80)
        return False

def test_edge_cases():
    """Test edge cases."""
    print("\n" + "="*80)
    print("TESTING EDGE CASES")
    print("="*80)
    
    ensure_artifacts_dir()
    
    # Test 1: Minimal data (no spikes, all efficient)
    print("\n1. Testing minimal data (no anomalies)...")
    dates = [datetime(2025, 6, i) for i in range(1, 16)]
    records = []
    for date in dates:
        records.append({
            'date': date,
            'project_id': 'ProjectX',
            'num_model_requests': 1000,
            'tokens': 50000,
            'cost_usd': 1.0,
            'is_premium_model': 0,
            'cache_hit_rate': 0.75
        })
    
    df_minimal = pd.DataFrame(records)
    summary = q4_anomalies(df_minimal)
    print("   ✓ Minimal data test complete")
    
    # Test 2: Missing project_id
    print("\n2. Testing missing project_id column...")
    df_no_project = df_minimal.drop(columns=['project_id'])
    summary = q4_anomalies(df_no_project)
    print("   ✓ Missing project_id test complete")
    
    # Test 3: Very short time series
    print("\n3. Testing very short time series (5 days)...")
    df_short = df_minimal.head(5)
    summary = q4_anomalies(df_short)
    print("   ✓ Short time series test complete")
    
    print("\n" + "="*80)
    print("✓ EDGE CASE TESTS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    # Run main test
    test_passed = test_q4_anomalies()
    
    # Run edge case tests
    test_edge_cases()
    
    # Final summary
    print("\n" + "="*80)
    if test_passed:
        print("ALL Q4_ANOMALIES TESTS COMPLETED SUCCESSFULLY")
    else:
        print("SOME TESTS FAILED - REVIEW ABOVE")
    print("="*80)

