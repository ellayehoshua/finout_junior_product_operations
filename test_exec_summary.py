#!/usr/bin/env python3
"""
Test script for the enhanced print_executive_summary function.

Demonstrates the compact, executive-friendly output with concrete numbers.
"""

from main import print_executive_summary, ensure_artifacts_dir

def test_executive_summary():
    """Test the executive summary with sample results."""
    
    print("Testing Enhanced Executive Summary Function")
    print("=" * 80)
    
    # Create sample results from each Q function (mimicking real output)
    
    # Q1: Cost Drivers
    q1_result = """[Q1] SPEND DRIVERS:
  • Top 5 projects account for $38,234 (76.5% of total spend)
  • ProjectA leads at $12,456 (24.9% share) - Volume-driven
  • ProjectB: $9,876 (19.8%) - Mix-driven (85% premium models)
  • Average cost/1k tokens: $0.0245 (org-wide)
  • 3 projects flagged as inefficient (>1.5x median cost)"""
    
    # Q2: Usage Shifts
    q2_result = """[Q2] USAGE PATTERNS:
  • 5 significant pattern shifts detected over 30 days
  • Volume spike: +47% on June 15 (15,234 → 22,401 requests)
  • Prompt inflation: Avg tokens/req grew +32% (48 → 63 tokens)
  • Cache utilization dropped 23pp (78% → 55%) on June 18
  • Model mix shift: GPT-4 share +18pp (35% → 53%)"""
    
    # Q3: Budget
    q3_result = """[Q3] BUDGET STATUS:
  • MTD spend: $32,450 of $50,000 budget (64.9% consumed)
  • Days elapsed: 20 of 30 (66.7% of month)
  • Pacing: 0.97 (slightly under pace) ✓
  • Run-rate: $1,856/day (last 7 days)
  • Projected EOM: $51,106 ⚠️ OVER by $1,106 (2.2%)
  • 2 projects at risk of exceeding allocated budgets"""
    
    # Q4: Anomalies
    q4_result = """[Q4] ANOMALIES & INEFFICIENCIES:
  • Detected 3 spike event(s) (Spike Index > 3.0)
  • Identified 2 inefficient project(s) (Inefficiency Index > 1.3)
  • Worst spike: 2025-06-16 (Index: 4.23)
  • Most inefficient: ProjectC (Index: 1.87, cost 2.3x median)"""
    
    # Ensure artifacts directory exists
    ensure_artifacts_dir()
    
    # Call the enhanced executive summary
    print("\nCalling print_executive_summary()...\n")
    print_executive_summary(q1_result, q2_result, q3_result, q4_result)
    
    # Test with missing data
    print("\n\n" + "="*80)
    print("Testing with partial data (Q3 and Q4 missing)...")
    print("="*80 + "\n")
    
    print_executive_summary(q1_result, q2_result, None, None)


if __name__ == "__main__":
    test_executive_summary()
    
    print("\n" + "="*80)
    print("✓ Executive Summary Test Complete")
    print("="*80)
    print("\nThe enhanced function provides:")
    print("  • Clear visual hierarchy with box-drawing characters")
    print("  • Concrete numbers highlighted throughout")
    print("  • Compact, stakeholder-friendly format")
    print("  • Auto-counts generated artifacts")
    print("  • Professional executive-level presentation")
    print("="*80 + "\n")

