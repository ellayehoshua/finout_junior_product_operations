"""
Refactoring script to split main.py into modular files.

This script extracts functions from the monolithic main.py and creates
organized module files.
"""

import re

print("="*80)
print("REFACTORING main.py INTO MODULES")
print("="*80)

# Read the original main.py
with open('main.py', 'r', encoding='utf-8') as f:
    content = f.read()
    lines = f.readlines()

# Reset file pointer and read lines
with open('main.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

def find_function_range(lines, func_name):
    """Find start and end line indices for a function."""
    start_idx = None
    end_idx = None
    indent_level = None
    
    for i, line in enumerate(lines):
        if line.strip().startswith(f'def {func_name}('):
            start_idx = i
            # Find the indentation level
            indent_level = len(line) - len(line.lstrip())
            continue
        
        if start_idx is not None and i > start_idx:
            # Check if we've reached the next function at same or lower indent
            if line.strip() and not line.strip().startswith('#'):
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= indent_level and (line.strip().startswith('def ') or line.strip().startswith('class ')):
                    end_idx = i
                    break
    
    if start_idx is not None and end_idx is None:
        end_idx = len(lines)
    
    return start_idx, end_idx

# Extract Q1
print("\n1. Extracting Q1 analysis...")
start, end = find_function_range(lines, 'q1_drivers')
if start and end:
    q1_content = ''.join(lines[start:end])
    
    q1_file = f"""\"\"\"
Q1 Analysis: Which teams/projects drive spend?

Identifies top cost drivers and classifies them by:
- Volume-driven (high usage, reasonable costs)
- Mix-driven (premium model usage)
- Inefficiency-driven (high unit costs)
\"\"\"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from config import ARTIFACTS_DIR
from utils import safe_divide, export_table, save_fig


{q1_content}"""
    
    with open('q1_analysis.py', 'w', encoding='utf-8') as f:
        f.write(q1_file)
    print(f"   ✓ Created q1_analysis.py ({end-start} lines)")

# Extract Q2
print("\n2. Extracting Q2 analysis...")
start, end = find_function_range(lines, 'q2_usage_shifts')
if start and end:
    q2_content = ''.join(lines[start:end])
    
    q2_file = f"""\"\"\"
Q2 Analysis: How are usage patterns changing over time?

Tracks daily metrics and detects shifts in:
- Request volume (WoW changes)
- Tokens per request (prompt inflation/deflation)
- Cache hit rates
- Model mix (premium vs standard)
\"\"\"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import (
    ARTIFACTS_DIR, REQUESTS_WOW_THRESHOLD, TOKENS_REQ_THRESHOLD,
    CACHE_IO_THRESHOLD, MIX_THRESHOLD
)
from utils import safe_divide, export_table, save_fig, format_axis_labels


{q2_content}"""
    
    with open('q2_analysis.py', 'w', encoding='utf-8') as f:
        f.write(q2_file)
    print(f"   ✓ Created q2_analysis.py ({end-start} lines)")

# Extract Q3
print("\n3. Extracting Q3 analysis...")
start, end = find_function_range(lines, 'q3_budget')
if start and end:
    q3_content = ''.join(lines[start:end])
    
    q3_file = f"""\"\"\"
Q3 Analysis: Are costs on-budget or at risk?

Tracks budget pacing and projects end-of-month spend:
- Burn-up chart vs linear budget
- Run-rate calculations
- Projected EOM costs
- Pacing index
- Risk contributor identification
\"\"\"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from config import ARTIFACTS_DIR, MONTHLY_BUDGET
from utils import safe_divide, export_table, save_fig


{q3_content}"""
    
    with open('q3_analysis.py', 'w', encoding='utf-8') as f:
        f.write(q3_file)
    print(f"   ✓ Created q3_analysis.py ({end-start} lines)")

# Extract Q4
print("\n4. Extracting Q4 analysis...")
start, end = find_function_range(lines, 'q4_anomalies')
if start and end:
    q4_content = ''.join(lines[start:end])
    
    # Also get analyze_q4_anomalies if it exists
    start2, end2 = find_function_range(lines, 'analyze_q4_anomalies')
    q4_legacy = ''
    if start2 and end2:
        q4_legacy = '\n\n' + ''.join(lines[start2:end2])
    
    q4_file = f"""\"\"\"
Q4 Analysis: Where are anomalies, spikes, or inefficiencies?

Detects unusual patterns using:
- Spike Index: Composite z-score (requests + tokens)
- Inefficiency Index: Weighted formula (cost ratio + premium share + cache penalty)
- Flags days with Spike Index > 3.0
- Flags projects with Inefficiency Index > 1.3
\"\"\"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import ARTIFACTS_DIR, BASELINE_DAYS
from utils import safe_divide, export_table, save_fig


{q4_content}{q4_legacy}"""
    
    with open('q4_analysis.py', 'w', encoding='utf-8') as f:
        f.write(q4_file)
    print(f"   ✓ Created q4_analysis.py ({end-start} lines)")

# Extract reporting
print("\n5. Extracting reporting...")
start, end = find_function_range(lines, 'print_executive_summary')
if start and end:
    reporting_content = ''.join(lines[start:end])
    
    reporting_file = f"""\"\"\"
Executive Summary Reporting.

Formats and prints professional, stakeholder-ready executive summary
with concrete numbers and visual hierarchy.
\"\"\"

import os
from config import ARTIFACTS_DIR


{reporting_content}"""
    
    with open('reporting.py', 'w', encoding='utf-8') as f:
        f.write(reporting_file)
    print(f"   ✓ Created reporting.py ({end-start} lines)")

# Create new main.py
print("\n6. Creating refactored main.py...")

new_main = """#!/usr/bin/env python3
\"\"\"
FinOps OpenAI Cost & Usage Analysis - Main Entry Point

Orchestrates the complete analysis pipeline:
1. Load data from Excel
2. Derive calculated metrics
3. Run Q1-Q4 analyses
4. Generate executive summary

All detailed logic is in separate modules for maintainability.
\"\"\"

import json

# Configuration
from config import DATA_FILE, ARTIFACTS_DIR

# Utilities
from utils import ensure_artifacts_dir

# Data loading
from data_loader import load_data, derive_metrics

# Analysis functions
from q1_analysis import q1_drivers
from q2_analysis import q2_usage_shifts
from q3_analysis import q3_budget
from q4_analysis import q4_anomalies

# Reporting
from reporting import print_executive_summary


def main():
    \"\"\"
    Main execution function - orchestrates the entire analysis pipeline.
    \"\"\"
    print("\\n" + "="*80)
    print("FINOPS OPENAI COST & USAGE ANALYSIS")
    print("="*80)
    
    # Ensure output directory exists
    ensure_artifacts_dir()
    
    # Phase 1: Load data
    df, schema_report = load_data(DATA_FILE)
    
    if df is None:
        print("❌ Failed to load data. Exiting.")
        return
    
    # Save schema report
    if schema_report:
        schema_path = f"{ARTIFACTS_DIR}/schema_report.json"
        # Convert datetime objects to strings for JSON serialization
        if schema_report.get('date_range'):
            date_range = schema_report['date_range']
            if date_range.get('min'):
                date_range['min'] = str(date_range['min'])
            if date_range.get('max'):
                date_range['max'] = str(date_range['max'])
        
        with open(schema_path, 'w') as f:
            json.dump(schema_report, f, indent=2)
        print(f"✓ Schema report saved: {schema_path}\\n")
    
    # Phase 2: Derive metrics
    df = derive_metrics(df)
    
    # Phase 3: Run analyses
    print("\\n" + "="*80)
    print("RUNNING ANALYSES (Q1-Q4)")
    print("="*80)
    
    q1_result = q1_drivers(df)
    q2_result = q2_usage_shifts(df)
    q3_result = q3_budget(df)
    q4_result = q4_anomalies(df)
    
    # Phase 4: Print executive summary
    print_executive_summary(q1_result, q2_result, q3_result, q4_result)
    
    print("\\n" + "="*80)
    print("✓ ANALYSIS PIPELINE COMPLETE")
    print("="*80)
    print(f"\\nAll outputs saved to: {ARTIFACTS_DIR}/")
    print("="*80 + "\\n")


if __name__ == "__main__":
    main()
"""

with open('main_refactored.py', 'w', encoding='utf-8') as f:
    f.write(new_main)

print(f"   ✓ Created main_refactored.py")

# Summary
print("\n" + "="*80)
print("REFACTORING COMPLETE")
print("="*80)
print("\nCreated modules:")
print("  ✓ config.py - Configuration constants")
print("  ✓ utils.py - Utility functions")
print("  ✓ data_loader.py - Data loading & metrics")
print("  ✓ q1_analysis.py - Q1: Cost drivers")
print("  ✓ q2_analysis.py - Q2: Usage patterns")
print("  ✓ q3_analysis.py - Q3: Budget tracking")
print("  ✓ q4_analysis.py - Q4: Anomalies")
print("  ✓ reporting.py - Executive summary")
print("  ✓ main_refactored.py - Main entry point (< 100 lines!)")
print("\nTo use:")
print("  1. Review the generated files")
print("  2. Backup original: mv main.py main_original.py")
print("  3. Activate new main: mv main_refactored.py main.py")
print("  4. Test: python main.py")
print("="*80 + "\n")
"""

