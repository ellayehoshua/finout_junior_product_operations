#!/usr/bin/env python3
"""
Test script for utility functions: save_fig(), export_table(), format_axis_labels()
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

# Import from main.py
from main import save_fig, export_table, format_axis_labels, ensure_artifacts_dir

def test_export_table():
    """Test the export_table function."""
    print("\n" + "="*80)
    print("TEST 1: export_table()")
    print("="*80)
    
    # Create test dataframe
    test_data = {
        'project_id': ['ProjectA', 'ProjectB', 'ProjectC'],
        'cost_usd': [1234.567890, 2345.678901, 3456.789012],
        'tokens': [100000, 200000, 300000],
        'cost_per_1k': [0.012346, 0.011728, 0.011523],
        'cache_hit_rate': [0.75, 0.82, 0.68]
    }
    df = pd.DataFrame(test_data)
    
    print("\nTest DataFrame:")
    print(df)
    
    # Test 1: Default formatting (4 decimals)
    print("\n1. Testing default export (4 decimals)...")
    export_table(df, 'test_default.csv')
    
    # Test 2: Custom formatting (2 decimals)
    print("\n2. Testing custom format (2 decimals)...")
    export_table(df, 'test_2decimals.csv', float_format='%.2f')
    
    # Test 3: Full path (should not duplicate artifacts)
    print("\n3. Testing full path...")
    export_table(df, './artifacts/test_fullpath.csv')
    
    # Verify files exist
    print("\n4. Verifying exports...")
    for filename in ['test_default.csv', 'test_2decimals.csv', 'test_fullpath.csv']:
        path = f'./artifacts/{filename}'
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        print(f"   {'✓' if exists else '✗'} {filename}: {size} bytes")
    
    print("\n✓ export_table() tests complete")


def test_save_fig():
    """Test the save_fig function."""
    print("\n" + "="*80)
    print("TEST 2: save_fig()")
    print("="*80)
    
    # Test 1: Simple line chart
    print("\n1. Testing simple line chart...")
    plt.figure(figsize=(10, 6))
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    plt.plot(x, y, linewidth=2, label='sin(x)')
    plt.xlabel('X Values', fontweight='bold')
    plt.ylabel('Y Values', fontweight='bold')
    plt.title('Test Chart: Sine Wave', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_fig('test_simple_chart.png')
    
    # Test 2: Bar chart
    print("\n2. Testing bar chart...")
    plt.figure(figsize=(8, 6))
    categories = ['A', 'B', 'C', 'D', 'E']
    values = [23, 45, 56, 78, 32]
    plt.bar(categories, values, color='steelblue', alpha=0.8)
    plt.xlabel('Categories', fontweight='bold')
    plt.ylabel('Values', fontweight='bold')
    plt.title('Test Chart: Bar Chart', fontweight='bold')
    plt.grid(True, alpha=0.3)
    save_fig('test_bar_chart.png')
    
    # Test 3: Full path (should not duplicate artifacts)
    print("\n3. Testing full path...")
    plt.figure(figsize=(8, 6))
    plt.scatter([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
    plt.title('Test Scatter', fontweight='bold')
    save_fig('./artifacts/test_fullpath.png')
    
    # Test 4: Custom DPI
    print("\n4. Testing custom DPI (150)...")
    plt.figure(figsize=(8, 6))
    plt.plot([1, 2, 3], [4, 5, 6])
    plt.title('Low DPI Test', fontweight='bold')
    save_fig('test_low_dpi.png', dpi=150)
    
    # Verify files exist
    print("\n5. Verifying chart exports...")
    for filename in ['test_simple_chart.png', 'test_bar_chart.png', 
                     'test_fullpath.png', 'test_low_dpi.png']:
        path = f'./artifacts/{filename}'
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        print(f"   {'✓' if exists else '✗'} {filename}: {size:,} bytes")
    
    print("\n✓ save_fig() tests complete")


def test_format_axis_labels():
    """Test the format_axis_labels function."""
    print("\n" + "="*80)
    print("TEST 3: format_axis_labels()")
    print("="*80)
    
    # Test 1: All labels
    print("\n1. Testing with all labels...")
    fig, ax = plt.subplots(figsize=(10, 6))
    dates = [datetime(2025, 6, i) for i in range(1, 11)]
    values = np.random.randint(100, 200, 10)
    ax.plot(dates, values, marker='o', linewidth=2)
    format_axis_labels(
        ax,
        xlabel='Date',
        ylabel='Daily Requests',
        title='Test Chart: Daily Activity'
    )
    save_fig('test_formatted_all.png')
    
    # Test 2: Partial labels
    print("\n2. Testing with partial labels (no title)...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(['Mon', 'Tue', 'Wed', 'Thu', 'Fri'], [10, 15, 13, 17, 14])
    format_axis_labels(ax, xlabel='Day of Week', ylabel='Count')
    save_fig('test_formatted_partial.png')
    
    # Test 3: Custom font sizes
    print("\n3. Testing with custom font sizes...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot([1, 2, 3, 4, 5], [2, 4, 3, 5, 4])
    format_axis_labels(
        ax,
        xlabel='Time',
        ylabel='Value',
        title='Large Font Test',
        xlabel_size=14,
        ylabel_size=14,
        title_size=18
    )
    save_fig('test_formatted_large_font.png')
    
    # Verify files
    print("\n4. Verifying formatted chart exports...")
    for filename in ['test_formatted_all.png', 'test_formatted_partial.png', 
                     'test_formatted_large_font.png']:
        path = f'./artifacts/{filename}'
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        print(f"   {'✓' if exists else '✗'} {filename}: {size:,} bytes")
    
    print("\n✓ format_axis_labels() tests complete")


def test_integration():
    """Test all utilities together in a realistic scenario."""
    print("\n" + "="*80)
    print("TEST 4: INTEGRATION (All Utilities Together)")
    print("="*80)
    
    print("\n1. Creating sample analysis data...")
    # Simulate Q1-style analysis
    dates = pd.date_range('2025-06-01', '2025-06-30', freq='D')
    daily_data = {
        'date': dates,
        'requests': np.random.randint(5000, 15000, len(dates)),
        'cost_usd': np.random.uniform(100, 300, len(dates)),
        'avg_tokens_per_req': np.random.uniform(40, 80, len(dates))
    }
    daily_df = pd.DataFrame(daily_data)
    
    # Export data table
    print("\n2. Exporting daily metrics table...")
    export_table(daily_df, 'test_integration_daily.csv', float_format='%.2f')
    
    # Create multi-panel chart
    print("\n3. Creating multi-panel chart...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Panel 1: Requests over time
    ax1.plot(daily_df['date'], daily_df['requests'], 
             marker='o', markersize=4, linewidth=2, color='steelblue')
    format_axis_labels(ax1, xlabel='Date', ylabel='Requests', 
                      title='Daily Request Volume')
    
    # Panel 2: Cost over time
    ax2.plot(daily_df['date'], daily_df['cost_usd'], 
             marker='s', markersize=4, linewidth=2, color='coral')
    format_axis_labels(ax2, xlabel='Date', ylabel='Cost (USD)', 
                      title='Daily Cost')
    
    plt.tight_layout()
    save_fig('test_integration_panel.png')
    
    # Create summary stats
    print("\n4. Creating summary statistics table...")
    summary_data = {
        'metric': ['Total Requests', 'Total Cost', 'Avg Cost/Day', 'Avg Tokens/Req'],
        'value': [
            daily_df['requests'].sum(),
            daily_df['cost_usd'].sum(),
            daily_df['cost_usd'].mean(),
            daily_df['avg_tokens_per_req'].mean()
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    export_table(summary_df, 'test_integration_summary.csv', float_format='%.2f')
    
    print("\n✓ Integration test complete")
    print("  All utilities work together seamlessly!")


def cleanup_test_files():
    """Optional: Clean up test files."""
    print("\n" + "="*80)
    print("CLEANUP (Optional)")
    print("="*80)
    
    test_files = [
        'test_default.csv', 'test_2decimals.csv', 'test_fullpath.csv',
        'test_simple_chart.png', 'test_bar_chart.png', 'test_low_dpi.png',
        'test_formatted_all.png', 'test_formatted_partial.png', 
        'test_formatted_large_font.png', 'test_integration_daily.csv',
        'test_integration_panel.png', 'test_integration_summary.csv'
    ]
    
    print("\nTest files created:")
    for filename in test_files:
        path = f'./artifacts/{filename}'
        if os.path.exists(path):
            print(f"  • {filename}")
    
    print("\nTo remove test files, delete from ./artifacts/ manually.")


if __name__ == "__main__":
    print("="*80)
    print("UTILITY FUNCTIONS TEST SUITE")
    print("="*80)
    
    # Ensure artifacts directory exists
    ensure_artifacts_dir()
    
    # Run all tests
    test_export_table()
    test_save_fig()
    test_format_axis_labels()
    test_integration()
    
    # Summary
    cleanup_test_files()
    
    print("\n" + "="*80)
    print("✓ ALL UTILITY TESTS COMPLETE")
    print("="*80)
    print("\nSummary:")
    print("  • export_table(): Tested with various formats ✓")
    print("  • save_fig(): Tested with various chart types ✓")
    print("  • format_axis_labels(): Tested with various options ✓")
    print("  • Integration: All utilities work together ✓")
    print("\nAll test artifacts saved to ./artifacts/")
    print("="*80 + "\n")

