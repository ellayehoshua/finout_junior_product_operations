"""
Executive Summary Reporting.

Formats and prints professional, stakeholder-ready executive summary
with concrete numbers and visual hierarchy.
"""

import os
from config import ARTIFACTS_DIR


def print_executive_summary(q1_result, q2_result, q3_result, q4_result):
    """
    Print compact executive summary with concrete numbers from all analyses.
    
    Displays a professional, stakeholder-ready recap with:
    - Key metrics and findings from each analysis question
    - Concrete numbers (costs, percentages, counts)
    - Actionable insights
    - Visual hierarchy for quick scanning
    
    Args:
        q1_result: String summary from q1_drivers()
        q2_result: String summary from q2_usage_shifts()
        q3_result: String summary from q3_budget()
        q4_result: String summary from q4_anomalies()
    """
    # Header
    print("\n" + "="*80)
    print(" " * 28 + "EXECUTIVE SUMMARY")
    print("="*80)
    print("\n📊 OpenAI Cost & Usage Analysis - Key Findings\n")
    
    # Q1: Cost Drivers
    print("┌─ [1] WHERE IS THE MONEY GOING?")
    if q1_result:
        # Print with indentation for clean hierarchy
        for line in q1_result.split('\n'):
            if line.strip():
                if line.startswith('[Q1]'):
                    continue  # Skip the header, we already have it
                print(f"│  {line}")
    else:
        print("│  ⚠️  Insufficient data")
    print("│")
    
    # Q2: Usage Patterns
    print("┌─ [2] HOW ARE USAGE PATTERNS CHANGING?")
    if q2_result:
        for line in q2_result.split('\n'):
            if line.strip():
                if line.startswith('[Q2]'):
                    continue
                print(f"│  {line}")
    else:
        print("│  ⚠️  Insufficient data")
    print("│")
    
    # Q3: Budget Status
    print("┌─ [3] ARE WE ON BUDGET?")
    if q3_result:
        for line in q3_result.split('\n'):
            if line.strip():
                if line.startswith('[Q3]'):
                    continue
                print(f"│  {line}")
    else:
        print("│  ⚠️  Insufficient data")
    print("│")
    
    # Q4: Anomalies
    print("┌─ [4] WHERE ARE THE PROBLEMS?")
    if q4_result:
        for line in q4_result.split('\n'):
            if line.strip():
                if line.startswith('[Q4]'):
                    continue
                print(f"│  {line}")
    else:
        print("│  ⚠️  Insufficient data")
    print("│")
    
    # Footer with action items
    print("└─" + "─"*78)
    print("\n📁 DELIVERABLES:")
    print("   • All charts and tables saved to ./artifacts/")
    print("   • Review detailed outputs for deeper insights")
    
    # Count artifacts
    if os.path.exists(ARTIFACTS_DIR):
        files = os.listdir(ARTIFACTS_DIR)
        csv_count = len([f for f in files if f.endswith('.csv')])
        png_count = len([f for f in files if f.endswith('.png')])
        print(f"   • Generated {csv_count} CSV tables + {png_count} PNG charts")
    
    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE")
    print("="*80 + "\n")

