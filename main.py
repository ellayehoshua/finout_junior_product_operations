"""
OpenAI Cost and Usage Analysis
Senior FinOps Data Engineer - Analysis Tool

This script reads OpenAI usage/cost dataset and produces metrics + charts for:
1. Which teams/projects drive spend
2. How usage patterns change over time
3. Are costs on-budget or at risk
4. Where anomalies, spikes, or inefficiencies happen
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure font - Poppins
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Poppins', 'Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

# Configure Seaborn style
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.1)

# ============================================================================
# CONSTANTS & THRESHOLDS
# ============================================================================

# Data source - Update these to your actual file/sheet names
DATA_FILE = "OpenAI Cost and Usage Data - June 2025.xlsx"
COST_SHEET = "Cost"      # Name of sheet/table with cost data
USAGE_SHEET = "Usage"    # Name of sheet/table with usage data

# Output directory
ARTIFACTS_DIR = "./artifacts"

# Anomaly detection thresholds
SPIKE_THRESHOLD_PCT = 0.30  # 30% WoW increase
SPIKE_Z_THRESHOLD = 3.0     # z-score threshold
TOKENS_PER_REQ_SHIFT_PCT = 0.20  # 20% shift threshold
IO_RATIO_SHIFT_PP = 0.20    # 20 percentage points
CACHE_CHANGE_PP = 0.20      # 20 percentage points
MODEL_MIX_SWING_PP = 0.20   # 20 percentage points
INEFFICIENCY_MULTIPLIER = 1.5  # 1.5x org median

# Budget parameters (example - adjust as needed)
MONTHLY_BUDGET = 50000  # USD
DAYS_IN_MONTH = 30

# Moving average window for trend confirmation
MA_WINDOW = 3

# Baseline window for z-score calculation
BASELINE_DAYS = 14

# Custom color palette
COLORS = ['#38B28E', '#FFD632', '#F78251', '#F379AC', '#7F7AFF', '#4CA2FF']
COLOR_PRIMARY = '#38B28E'    # Green
COLOR_WARNING = '#FFD632'    # Yellow  
COLOR_DANGER = '#F78251'     # Orange
COLOR_ACCENT = '#F379AC'     # Pink
COLOR_INFO = '#7F7AFF'       # Purple
COLOR_SECONDARY = '#4CA2FF'  # Blue

# Set Seaborn color palette
sns.set_palette(COLORS)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_artifacts_dir():
    """Create artifacts directory if it doesn't exist."""
    Path(ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)
    print(f"✓ Artifacts directory ready: {ARTIFACTS_DIR}")


def safe_divide(numerator, denominator, fill_value=0):
    """Safely divide two arrays/series, handling division by zero."""
    return np.where(denominator != 0, numerator / denominator, fill_value)


def to_snake_case(name):
    """
    Convert column name to snake_case.
    
    Args:
        name: Original column name
        
    Returns:
        snake_case version of the name
    """
    import re
    # Replace spaces and hyphens with underscores
    name = re.sub(r'[\s\-]+', '_', str(name))
    # Insert underscore before uppercase letters (camelCase -> snake_case)
    name = re.sub(r'(?<!^)(?=[A-Z])', '_', name)
    # Convert to lowercase
    name = name.lower()
    # Remove duplicate underscores
    name = re.sub(r'_+', '_', name)
    # Remove leading/trailing underscores
    name = name.strip('_')
    return name


def save_fig(path, dpi=300):
    """
    Save current matplotlib figure with consistent formatting.
    
    Ensures:
    - High-quality DPI (default 300)
    - Tight bounding box
    - Consistent styling across all charts
    - Proper path handling (adds ./artifacts if needed)
    
    Args:
        path: Output path (auto-adds ./artifacts/ if not present)
        dpi: Resolution (default 300)
    """
    # Ensure path includes artifacts directory
    if not path.startswith(ARTIFACTS_DIR):
        path = f"{ARTIFACTS_DIR}/{path}"
    
    # Save with high quality
    plt.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {path}")


def export_table(df, path, float_format='%.4f'):
    """
    Export DataFrame to CSV with consistent formatting.
    
    Features:
    - Consistent float formatting
    - Clean column headers
    - UTF-8 encoding
    - Proper path handling (adds ./artifacts if needed)
    
    Args:
        df: DataFrame to export
        path: Output path (auto-adds ./artifacts/ if not present)
        float_format: Format string for float columns (default '%.4f')
    """
    # Ensure path includes artifacts directory
    if not path.startswith(ARTIFACTS_DIR):
        path = f"{ARTIFACTS_DIR}/{path}"
    
    # Export with consistent formatting
    df.to_csv(path, index=False, float_format=float_format)
    print(f"  ✓ Saved: {path} ({len(df)} rows)")


def format_axis_labels(ax, xlabel=None, ylabel=None, title=None, 
                       xlabel_size=11, ylabel_size=11, title_size=13):
    """
    Apply consistent formatting to matplotlib axes.
    
    Sets:
    - Bold labels with appropriate font sizes
    - Grid with transparency
    - Rotated x-axis labels if needed
    
    Args:
        ax: Matplotlib axis object
        xlabel: X-axis label (optional)
        ylabel: Y-axis label (optional)
        title: Chart title (optional)
        xlabel_size: Font size for x-axis label
        ylabel_size: Font size for y-axis label
        title_size: Font size for title
    """
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=xlabel_size, fontweight='bold')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=ylabel_size, fontweight='bold')
    if title:
        ax.set_title(title, fontsize=title_size, fontweight='bold')
    
    # Add grid for readability
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels for date axes
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')


def load_data(file_path):
    """
    Load OpenAI cost and usage data from Excel file with TWO separate tables.
    
    Features:
    - Loads separate COST and USAGE tables
    - Merges them on project_id and time period
    - Maps columns to expected format
    - Converts column names to snake_case
    - Parses date columns to datetime
    - Reports found vs missing expected columns
    - Returns merged DataFrame and schema_report dict
    
    Args:
        file_path: Path to Excel file
        
    Returns:
        tuple: (DataFrame, schema_report dict) or (None, None) if error
    """
    # Expected columns for a complete dataset
    EXPECTED_COLUMNS = [
        'date', 'project_id', 'user_id', 'api_key_id', 'model', 
        'service_tier', 'input_tokens', 'output_tokens', 
        'input_cached_tokens', 'input_uncached_tokens', 
        'num_model_requests', 'cost_usd', 'is_premium_model'
    ]
    
    try:
        print(f"\n{'='*80}")
        print(f"LOADING DATA (COST + USAGE TABLES)")
        print(f"{'='*80}")
        print(f"File: {file_path}")
        
        # Read Excel file
        xl_file = pd.ExcelFile(file_path)
        print(f"  Found {len(xl_file.sheet_names)} sheet(s): {', '.join(xl_file.sheet_names)}")
        
        # ====================================================================
        # STEP 1: LOAD COST TABLE
        # ====================================================================
        print(f"\n  [1/3] Loading COST table...")
        
        # Try to find cost sheet
        cost_sheet_candidates = [COST_SHEET, 'cost', 'Cost', 'COST', 'costs', 'Costs']
        cost_sheet_name = None
        for candidate in cost_sheet_candidates:
            if candidate in xl_file.sheet_names:
                cost_sheet_name = candidate
                break
        
        if cost_sheet_name is None:
            print(f"  ❌ ERROR: Cost sheet not found. Tried: {', '.join(cost_sheet_candidates)}")
            print(f"  Available sheets: {', '.join(xl_file.sheet_names)}")
            return None, None
        
        cost_df = pd.read_excel(file_path, sheet_name=cost_sheet_name)
        print(f"    ✓ Loaded cost table: {len(cost_df):,} rows × {len(cost_df.columns)} columns")
        
        # Normalize column names
        cost_df.columns = [to_snake_case(col) for col in cost_df.columns]
        
        # ====================================================================
        # STEP 2: LOAD USAGE TABLE
        # ====================================================================
        print(f"\n  [2/3] Loading USAGE table...")
        
        # Try to find usage sheet
        usage_sheet_candidates = [USAGE_SHEET, 'usage', 'Usage', 'USAGE', 'usages', 'Usages']
        usage_sheet_name = None
        for candidate in usage_sheet_candidates:
            if candidate in xl_file.sheet_names:
                usage_sheet_name = candidate
                break
        
        if usage_sheet_name is None:
            print(f"  ❌ ERROR: Usage sheet not found. Tried: {', '.join(usage_sheet_candidates)}")
            print(f"  Available sheets: {', '.join(xl_file.sheet_names)}")
            return None, None
        
        usage_df = pd.read_excel(file_path, sheet_name=usage_sheet_name)
        print(f"    ✓ Loaded usage table: {len(usage_df):,} rows × {len(usage_df.columns)} columns")
        
        # Normalize column names
        usage_df.columns = [to_snake_case(col) for col in usage_df.columns]
        
        # ====================================================================
        # STEP 3: MERGE COST + USAGE TABLES
        # ====================================================================
        print(f"\n  [3/3] Merging cost and usage data...")
        
        # Parse dates in both tables
        for df_name, df in [('cost', cost_df), ('usage', usage_df)]:
            # Try to find time column - prefer ISO format
            time_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
            if time_cols:
                # Prefer: start_time_iso > start_time > end_time_iso > end_time > first time col
                time_col = None
                for candidate in ['start_time_iso', 'start_time', 'end_time_iso', 'end_time']:
                    if candidate in time_cols:
                        time_col = candidate
                        break
                if time_col is None:
                    time_col = time_cols[0]
                
                # Try different date parsing strategies
                # 1. Try ISO format first (string format)
                if 'iso' in time_col.lower() or df[time_col].dtype == 'object':
                    df['date'] = pd.to_datetime(df[time_col], errors='coerce')
                # 2. Try as Unix timestamp (milliseconds for int/float)
                elif df[time_col].dtype in ['int64', 'float64']:
                    df['date'] = pd.to_datetime(df[time_col], unit='ms', errors='coerce')
                # 3. Default: let pandas infer
                else:
                    df['date'] = pd.to_datetime(df[time_col], errors='coerce')
                
                print(f"    ✓ Parsed {df_name} dates from '{time_col}'")
        
        # Map cost columns
        cost_df = cost_df.rename(columns={
            'amount_value': 'cost_usd',
            'amount_currency': 'currency'
        })
        
        # Merge on project_id and date (using nearest time match)
        # For simplicity, we'll use left join from usage to cost
        df = usage_df.merge(
            cost_df[['project_id', 'date', 'cost_usd'] + 
                    ([col for col in cost_df.columns if col in ['organization_id', 'project_name', 'organization_name']])],
            on=['project_id', 'date'],
            how='left'
        )
        
        print(f"    ✓ Merged: {len(df):,} rows × {len(df.columns)} columns")
        
        # ====================================================================
        # STEP 4: FINALIZE MERGED DATA
        # ====================================================================
        
        # Create 'day' column (date only, no time)
        if 'date' in df.columns:
            df['day'] = pd.to_datetime(df['date']).dt.date
            
            # Print date range
            if df['date'].notna().any():
                min_date = df['date'].min()
                max_date = df['date'].max()
                print(f"    Date range: {min_date.date()} to {max_date.date()}")
        
        # Fill missing cost_usd with 0 (if usage row has no matching cost)
        if 'cost_usd' in df.columns:
            df['cost_usd'] = df['cost_usd'].fillna(0)
        
        # Schema report: found vs missing columns
        found_columns = [col for col in EXPECTED_COLUMNS if col in df.columns]
        missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]
        extra_columns = [col for col in df.columns if col not in EXPECTED_COLUMNS and col != 'day']
        
        print(f"  Found columns ({len(found_columns)}/{len(EXPECTED_COLUMNS)}):")
        if found_columns:
            for col in found_columns:
                # Show sample non-null count
                non_null = df[col].notna().sum()
                pct = 100 * non_null / len(df)
                print(f"    ✓ {col:<25} ({non_null:,} rows, {pct:.1f}% filled)")
        else:
            print(f"    (none)")
        
        if missing_columns:
            print(f"\n  Missing columns ({len(missing_columns)}):")
            for col in missing_columns:
                print(f"    ✗ {col}")
        
        if extra_columns:
            print(f"\n  Extra columns ({len(extra_columns)}) - not in expected schema:")
            for col in extra_columns[:10]:  # Show first 10
                print(f"    + {col}")
            if len(extra_columns) > 10:
                print(f"    ... and {len(extra_columns) - 10} more")
        
        # Build schema report dictionary
        schema_report = {
            'file_path': file_path,
            'cost_sheet': cost_sheet_name,
            'usage_sheet': usage_sheet_name,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'found_columns': found_columns,
            'missing_columns': missing_columns,
            'extra_columns': extra_columns,
            'has_date_column': 'date' in df.columns,
            'date_column': 'date',
            'date_range': {
                'min': df['date'].min() if 'date' in df.columns and df['date'].notna().any() else None,
                'max': df['date'].max() if 'date' in df.columns and df['date'].notna().any() else None
            } if 'date' in df.columns else None
        }
        
        print(f"\n  {'='*76}")
        print(f"  ✓ Data loaded and merged successfully!")
        print(f"  {'='*76}\n")
        
        return df, schema_report
    
    except FileNotFoundError:
        print(f"❌ ERROR: File not found: {file_path}")
        return None, None
    except Exception as e:
        print(f"❌ ERROR loading data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def check_required_columns(df, required_cols):
    """
    Check if required columns exist in dataframe.
    
    Args:
        df: Input dataframe
        required_cols: List of required column names
        
    Returns:
        tuple: (missing_cols, available_cols)
    """
    missing = [col for col in required_cols if col not in df.columns]
    available = [col for col in required_cols if col in df.columns]
    return missing, available


def derive_metrics(df):
    """
    Derive calculated metrics from base columns with comprehensive stats reporting.
    
    Features:
    - Calculates: tokens, tokens_per_req, io_ratio, cache_hit_rate, cost_per_1k, cost_per_req
    - Safe division with zero handling
    - Intelligent NA filling
    - MA3 (3-day moving average) for tokens_per_req by date
    - Detailed statistics report (mean, median, non-null counts)
    
    Args:
        df: Input dataframe with base columns
        
    Returns:
        DataFrame with derived metrics added
    """
    print("\n" + "="*80)
    print("DERIVING CALCULATED METRICS")
    print("="*80)
    
    # Work on a copy to avoid modifying original
    df = df.copy()
    derived_cols = []
    
    # Ensure date column is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # ========================================================================
    # 1. TOTAL TOKENS
    # ========================================================================
    if 'input_tokens' in df.columns and 'output_tokens' in df.columns:
        df['tokens'] = df['input_tokens'].fillna(0) + df['output_tokens'].fillna(0)
        # Replace 0 with NaN where both inputs are NaN (to distinguish from true 0)
        mask_both_na = df['input_tokens'].isna() & df['output_tokens'].isna()
        df.loc[mask_both_na, 'tokens'] = np.nan
        derived_cols.append('tokens')
        print("  ✓ tokens = input_tokens + output_tokens")
    
    # ========================================================================
    # 2. TOKENS PER REQUEST
    # ========================================================================
    if 'tokens' in df.columns and 'num_model_requests' in df.columns:
        df['tokens_per_req'] = safe_divide(
            df['tokens'], 
            df['num_model_requests'], 
            fill_value=np.nan  # Use NaN instead of 0 for invalid divisions
        )
        # Replace NaN with 0 only where tokens exist but requests are 0
        mask_zero_requests = (df['tokens'].notna()) & (df['num_model_requests'] == 0)
        df.loc[mask_zero_requests, 'tokens_per_req'] = 0
        
        derived_cols.append('tokens_per_req')
        print("  ✓ tokens_per_req = tokens / num_model_requests")
    
    # ========================================================================
    # 3. INPUT/OUTPUT RATIO
    # ========================================================================
    if 'output_tokens' in df.columns and 'input_tokens' in df.columns:
        df['io_ratio'] = safe_divide(
            df['output_tokens'], 
            df['input_tokens'], 
            fill_value=np.nan
        )
        # Replace extreme outliers (cap at 100 for reasonable ratios)
        df.loc[df['io_ratio'] > 100, 'io_ratio'] = np.nan
        
        derived_cols.append('io_ratio')
        print("  ✓ io_ratio = output_tokens / input_tokens")
    
    # ========================================================================
    # 4. CACHE HIT RATE
    # ========================================================================
    if 'input_cached_tokens' in df.columns and 'input_uncached_tokens' in df.columns:
        total_input = df['input_cached_tokens'].fillna(0) + df['input_uncached_tokens'].fillna(0)
        df['cache_hit_rate'] = safe_divide(
            df['input_cached_tokens'].fillna(0), 
            total_input, 
            fill_value=0  # 0 cache rate when no input tokens
        )
        # Ensure rate is between 0 and 1
        df['cache_hit_rate'] = df['cache_hit_rate'].clip(0, 1)
        
        derived_cols.append('cache_hit_rate')
        print("  ✓ cache_hit_rate = input_cached_tokens / (cached + uncached)")
    
    # ========================================================================
    # 5. COST PER 1K TOKENS
    # ========================================================================
    if 'cost_usd' in df.columns and 'tokens' in df.columns:
        df['cost_per_1k'] = safe_divide(
            1000 * df['cost_usd'], 
            df['tokens'], 
            fill_value=np.nan
        )
        # Cap unreasonable values (> $1000 per 1k tokens is likely an error)
        df.loc[df['cost_per_1k'] > 1000, 'cost_per_1k'] = np.nan
        
        derived_cols.append('cost_per_1k')
        print("  ✓ cost_per_1k = 1000 * cost_usd / tokens")
    
    # ========================================================================
    # 6. COST PER REQUEST
    # ========================================================================
    if 'cost_usd' in df.columns and 'num_model_requests' in df.columns:
        df['cost_per_req'] = safe_divide(
            df['cost_usd'], 
            df['num_model_requests'], 
            fill_value=np.nan
        )
        # Cap unreasonable values
        df.loc[df['cost_per_req'] > 100, 'cost_per_req'] = np.nan
        
        derived_cols.append('cost_per_req')
        print("  ✓ cost_per_req = cost_usd / num_model_requests")
    
    # ========================================================================
    # 7. PREMIUM MODEL FLAG (if not present)
    # ========================================================================
    if 'is_premium_model' not in df.columns and 'model' in df.columns:
        premium_keywords = ['gpt-4', 'claude-3-opus', 'claude-3.5-sonnet', 'o1', 'o3']
        df['is_premium_model'] = df['model'].str.lower().str.contains(
            '|'.join(premium_keywords), 
            na=False
        )
        print("  ✓ is_premium_model inferred from model name")
    
    # ========================================================================
    # 8. MA3 FOR TOKENS_PER_REQ BY DATE
    # ========================================================================
    if 'tokens_per_req' in df.columns and 'date' in df.columns:
        # Sort by date for proper moving average calculation
        df_sorted = df.sort_values('date')
        
        # Group by date and calculate weighted average tokens_per_req
        if 'num_model_requests' in df.columns:
            # Weighted average by number of requests
            daily_tokens = df_sorted.groupby('date').apply(
                lambda x: safe_divide(
                    (x['tokens_per_req'] * x['num_model_requests']).sum(),
                    x['num_model_requests'].sum(),
                    fill_value=np.nan
                )
            )
        else:
            # Simple average if no request counts
            daily_tokens = df_sorted.groupby('date')['tokens_per_req'].mean()
        
        # Calculate 3-day moving average
        ma3_series = daily_tokens.rolling(window=3, min_periods=1).mean()
        
        # Map back to original dataframe
        date_to_ma3 = ma3_series.to_dict()
        df['tokens_per_req_ma3'] = df['date'].map(date_to_ma3)
        
        derived_cols.append('tokens_per_req_ma3')
        print("  ✓ tokens_per_req_ma3 = 3-day moving average by date")
    
    # ========================================================================
    # STATISTICS REPORT
    # ========================================================================
    print("\n" + "-"*80)
    print("DERIVED METRICS STATISTICS")
    print("-"*80)
    
    stats_data = []
    for col in derived_cols:
        if col in df.columns:
            non_null_count = df[col].notna().sum()
            total_count = len(df)
            pct_filled = 100 * non_null_count / total_count if total_count > 0 else 0
            
            # Calculate stats only on non-null values
            if non_null_count > 0:
                mean_val = df[col].mean()
                median_val = df[col].median()
                min_val = df[col].min()
                max_val = df[col].max()
            else:
                mean_val = median_val = min_val = max_val = np.nan
            
            stats_data.append({
                'metric': col,
                'non_null': non_null_count,
                'pct_filled': pct_filled,
                'mean': mean_val,
                'median': median_val,
                'min': min_val,
                'max': max_val
            })
    
    # Print statistics table
    if stats_data:
        print(f"\n{'Metric':<25} {'Non-Null':>10} {'Fill %':>8} {'Mean':>12} {'Median':>12}")
        print("-"*80)
        for stat in stats_data:
            print(f"{stat['metric']:<25} {stat['non_null']:>10,} {stat['pct_filled']:>7.1f}% "
                  f"{stat['mean']:>12.4f} {stat['median']:>12.4f}")
        
        print("\n" + "-"*80)
        print(f"Total rows: {len(df):,}")
        print(f"Derived metrics: {len(derived_cols)}")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("✓ Metric derivation complete")
    print("="*80 + "\n")
    
    return df


# ============================================================================
# Q1: WHICH TEAMS/PROJECTS DRIVE SPEND?
# ============================================================================

def q1_drivers(df):
    """
    Analyze top spending projects with driver classification.
    
    Features:
    - Top 5 projects by sum(cost_usd)
    - Columns: cost, % of total, tokens, requests, cost/1k, % premium tokens, cache_hit_rate
    - Classification: Volume-driven, Mix-driven, or Inefficiency-driven
    - Outputs: q1_top5_projects.csv and q1_top5_bar.png
    - Returns: Executive summary string (2-4 bullets with numbers)
    
    Args:
        df: Input dataframe with cost and project data
        
    Returns:
        str: Executive summary with key findings
    """
    print("\n" + "="*80)
    print("Q1: WHICH TEAMS/PROJECTS DRIVE SPEND?")
    print("="*80)
    
    # Check required columns
    required_cols = ['project_id', 'cost_usd']
    missing = [col for col in required_cols if col not in df.columns]
    
    if missing:
        print(f"❌ Insufficient data for Q1. Missing columns: {missing}")
        return "Q1: Insufficient data - missing required columns"
    
    # Work on a copy
    df = df.copy()
    
    # ========================================================================
    # STEP 1: AGGREGATE BY PROJECT
    # ========================================================================
    
    # Prepare aggregation dictionary
    agg_dict = {'cost_usd': 'sum'}
    
    # Add optional columns
    if 'tokens' in df.columns:
        agg_dict['tokens'] = 'sum'
    if 'num_model_requests' in df.columns:
        agg_dict['num_model_requests'] = 'sum'
    
    # Calculate premium token percentage
    if 'tokens' in df.columns and 'is_premium_model' in df.columns:
        df['premium_tokens'] = df['tokens'] * df['is_premium_model'].astype(int)
        agg_dict['premium_tokens'] = 'sum'
    
    # Weighted cache hit rate
    if 'cache_hit_rate' in df.columns and 'tokens' in df.columns:
        df['cache_weighted'] = df['cache_hit_rate'] * df['tokens']
        agg_dict['cache_weighted'] = 'sum'
    
    # Aggregate by project
    project_summary = df.groupby('project_id').agg(agg_dict).reset_index()
    
    # ========================================================================
    # STEP 2: CALCULATE DERIVED METRICS
    # ========================================================================
    
    total_cost = project_summary['cost_usd'].sum()
    
    # Percentage of total cost
    project_summary['pct_of_total'] = 100 * project_summary['cost_usd'] / total_cost
    
    # Cost per 1k tokens
    if 'tokens' in project_summary.columns:
        project_summary['cost_per_1k'] = safe_divide(
            1000 * project_summary['cost_usd'],
            project_summary['tokens'],
            fill_value=np.nan
        )
    else:
        project_summary['cost_per_1k'] = np.nan
    
    # Percentage premium tokens
    if 'premium_tokens' in project_summary.columns and 'tokens' in project_summary.columns:
        project_summary['pct_premium_tokens'] = 100 * safe_divide(
            project_summary['premium_tokens'],
            project_summary['tokens'],
            fill_value=0
        )
    else:
        project_summary['pct_premium_tokens'] = 0
    
    # Average cache hit rate (weighted)
    if 'cache_weighted' in project_summary.columns and 'tokens' in project_summary.columns:
        project_summary['cache_hit_rate'] = safe_divide(
            project_summary['cache_weighted'],
            project_summary['tokens'],
            fill_value=0
        )
    else:
        project_summary['cache_hit_rate'] = 0
    
    # ========================================================================
    # STEP 3: SELECT TOP 5 PROJECTS
    # ========================================================================
    
    project_summary = project_summary.sort_values('cost_usd', ascending=False)
    top5 = project_summary.head(5).copy()
    
    # ========================================================================
    # STEP 4: CLASSIFY EACH PROJECT BY DRIVER TYPE
    # ========================================================================
    
    # Calculate organization-wide benchmarks
    org_median_cost_per_1k = project_summary['cost_per_1k'].median()
    org_avg_premium_pct = project_summary['pct_premium_tokens'].mean()
    
    # Define thresholds for classification
    VOLUME_THRESHOLD = 30  # Top 30% of requests/tokens = high volume
    MIX_THRESHOLD = org_avg_premium_pct + 15  # 15pp above average premium usage
    INEFFICIENCY_THRESHOLD = 1.3  # 1.3x org median cost/1k
    
    def classify_driver(row, org_median, all_projects):
        """Classify project as Volume, Mix, or Inefficiency driven."""
        
        # Get percentile ranks for volume (tokens)
        if 'tokens' in row and pd.notna(row.get('tokens', np.nan)):
            volume_percentile = (all_projects['tokens'] < row['tokens']).sum() / len(all_projects) * 100
        else:
            volume_percentile = 50  # Default to median
        
        cost_ratio = row['cost_per_1k'] / org_median if pd.notna(row['cost_per_1k']) and org_median > 0 else 1.0
        premium_pct = row.get('pct_premium_tokens', 0)
        
        # Classification logic (in priority order)
        
        # 1. Inefficiency-driven: High unit cost (primary indicator)
        if cost_ratio >= INEFFICIENCY_THRESHOLD:
            return 'Inefficiency-driven'
        
        # 2. Mix-driven: High premium usage with above-avg cost
        if premium_pct >= MIX_THRESHOLD and cost_ratio > 1.0:
            return 'Mix-driven'
        
        # 3. Volume-driven: High volume with reasonable costs
        if volume_percentile >= VOLUME_THRESHOLD:
            return 'Volume-driven'
        
        # 4. Mixed: Balanced profile
        if premium_pct > org_avg_premium_pct:
            return 'Mix-driven'
        else:
            return 'Volume-driven'
    
    # Apply classification
    top5['driver_type'] = top5.apply(
        lambda row: classify_driver(row, org_median_cost_per_1k, project_summary),
        axis=1
    )
    
    # ========================================================================
    # STEP 5: PREPARE OUTPUT TABLE
    # ========================================================================
    
    # Select and rename columns for export
    export_cols = ['project_id', 'cost_usd', 'pct_of_total']
    
    if 'tokens' in top5.columns:
        export_cols.append('tokens')
    if 'num_model_requests' in top5.columns:
        export_cols.append('num_model_requests')
    if 'cost_per_1k' in top5.columns:
        export_cols.append('cost_per_1k')
    if 'pct_premium_tokens' in top5.columns:
        export_cols.append('pct_premium_tokens')
    if 'cache_hit_rate' in top5.columns:
        export_cols.append('cache_hit_rate')
    
    export_cols.append('driver_type')
    
    export_table = top5[export_cols].copy()
    
    # Rename for clarity
    rename_map = {
        'num_model_requests': 'requests',
        'pct_of_total': 'pct_of_total_cost'
    }
    export_table.rename(columns=rename_map, inplace=True)
    
    # ========================================================================
    # STEP 6: SAVE CSV
    # ========================================================================
    
    csv_path = f"{ARTIFACTS_DIR}/q1_top5_projects.csv"
    export_table.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"✓ Saved table: {csv_path}")
    
    # ========================================================================
    # STEP 7: CREATE BAR CHART
    # ========================================================================
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Color-code by driver type
    color_map = {
        'Volume-driven': COLOR_SECONDARY,
        'Mix-driven': COLOR_WARNING,
        'Inefficiency-driven': COLOR_DANGER
    }
    
    # Create data for seaborn
    top5_sorted = top5.sort_values('cost_usd', ascending=True)
    colors = [color_map.get(dt, 'gray') for dt in top5_sorted['driver_type']]
    
    # Create horizontal bar chart with Seaborn
    sns.barplot(
        data=top5_sorted,
        y='project_id',
        x='cost_usd',
        palette=colors,
        ax=ax,
        alpha=0.8
    )
    
    # Add driver type to y-labels
    labels = [f"{pid}\n({dt})" for pid, dt in zip(top5_sorted['project_id'], top5_sorted['driver_type'])]
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Cost (USD)', fontsize=11, fontweight='bold')
    ax.set_ylabel('')
    ax.set_title('Top 5 Projects by Cost (Colored by Driver Type)', fontsize=13, fontweight='bold')
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(top5_sorted.iterrows()):
        cost = row['cost_usd']
        pct = row['pct_of_total']
        label_text = f'  ${cost:,.0f} ({pct:.1f}%)'
        ax.text(cost, i, label_text, va='center', fontsize=9, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_map['Volume-driven'], alpha=0.8, label='Volume-driven'),
        Patch(facecolor=color_map['Mix-driven'], alpha=0.8, label='Mix-driven'),
        Patch(facecolor=color_map['Inefficiency-driven'], alpha=0.8, label='Inefficiency-driven')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    plt.tight_layout()
    chart_path = f"{ARTIFACTS_DIR}/q1_top5_bar.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved chart: {chart_path}")
    
    # ========================================================================
    # ADDITIONAL Q1 CHARTS
    # ========================================================================
    
    # Chart 2: Pie chart of cost distribution
    fig, ax = plt.subplots(figsize=(14, 11))
    
    # Prepare data: Top 5 + "Others"
    others_cost = total_cost - top5['cost_usd'].sum()
    pie_labels = list(top5['project_id']) + ['Others']
    pie_values = list(top5['cost_usd']) + [others_cost]
    pie_colors = COLORS[:len(top5)] + ['#CCCCCC']
    
    # Calculate percentages
    pie_percentages = [(v / total_cost) * 100 for v in pie_values]
    
    # Create pie chart with external labels
    wedges, texts = ax.pie(
        pie_values,
        labels=None,  # We'll add custom labels
        colors=pie_colors,
        startangle=90,
        wedgeprops={'linewidth': 2, 'edgecolor': 'white'}
    )
    
    # Add labels with leader lines - smart positioning to avoid overlap
    bbox_props = dict(boxstyle="round,pad=0.4", facecolor='white', edgecolor='gray', alpha=0.9, linewidth=1.5)
    
    # Collect label positions by side (left/right) and angle for smart spacing
    label_positions = []
    for i, (wedge, label, pct) in enumerate(zip(wedges, pie_labels, pie_percentages)):
        ang = (wedge.theta2 - wedge.theta1) / 2. + wedge.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        
        # Determine distance based on slice size
        if pct >= 9:  # Large slices - closer to pie
            distance_x = 1.15
            distance_y = 1.15
        else:  # Small slices - further away
            distance_x = 1.5
            distance_y = 1.5
        
        label_positions.append({
            'label': label,
            'pct': pct,
            'wedge': wedge,
            'ang': ang,
            'x': x,
            'y': y,
            'distance_x': distance_x,
            'distance_y': distance_y,
            'side': 'right' if x > 0 else 'left'
        })
    
    # Sort by side and then by y position (descending for better spacing)
    left_labels = sorted([p for p in label_positions if p['side'] == 'left'], key=lambda p: p['y'], reverse=True)
    right_labels = sorted([p for p in label_positions if p['side'] == 'right'], key=lambda p: p['y'], reverse=True)
    
    # Function to adjust y positions to avoid overlap - improved algorithm
    def adjust_y_positions(labels, min_spacing=0.22):
        if len(labels) <= 1:
            return labels
        
        # Multiple passes to resolve all conflicts
        max_iterations = 10
        for iteration in range(max_iterations):
            adjusted = False
            
            # Check all pairs for overlap
            for i in range(len(labels)):
                for j in range(len(labels)):
                    if i == j:
                        continue
                    
                    # Calculate actual y positions
                    y_i = labels[i]['distance_y'] * labels[i]['y']
                    y_j = labels[j]['distance_y'] * labels[j]['y']
                    
                    # Check if too close
                    if abs(y_i - y_j) < min_spacing:
                        adjusted = True
                        
                        # Move the label that's easier to move (higher index, further from pie)
                        if i > j:
                            # Push i away from j
                            if y_i > y_j:
                                target_y = y_j + min_spacing
                            else:
                                target_y = y_j - min_spacing
                            
                            # Update distance
                            if labels[i]['y'] != 0:
                                labels[i]['distance_y'] = abs(target_y / labels[i]['y'])
                        else:
                            # Push j away from i
                            if y_j > y_i:
                                target_y = y_i + min_spacing
                            else:
                                target_y = y_i - min_spacing
                            
                            # Update distance
                            if labels[j]['y'] != 0:
                                labels[j]['distance_y'] = abs(target_y / labels[j]['y'])
            
            # If no adjustments made, we're done
            if not adjusted:
                break
        
        # Clamp y positions to stay within bounds
        for label in labels:
            y_pos = label['distance_y'] * label['y']
            # Keep labels between -0.9 and 0.9 to avoid title and bottom
            if y_pos > 0.9:
                label['distance_y'] = 0.9 / label['y'] if label['y'] > 0 else label['distance_y']
            elif y_pos < -0.9:
                label['distance_y'] = 0.9 / abs(label['y']) if label['y'] < 0 else label['distance_y']
        
        return labels
    
    # Adjust positions with better spacing
    left_labels = adjust_y_positions(left_labels, min_spacing=0.25)
    right_labels = adjust_y_positions(right_labels, min_spacing=0.25)
    
    # Combine back
    all_labels = left_labels + right_labels
    
    # Manual position overrides based on user requirements
    manual_positions = {
        'Others': (-0.8, 0.85),      # Top left - first position
        'proj_4': (-0.6, 0.75),      # Top left - second position  
        'proj_3': (-0.4, 0.70),      # Top left - third position
        'proj_2': (0.85, 0.75)       # Top right - separate position
    }
    
    # Draw annotations
    for pos in all_labels:
        x = pos['x']
        y = pos['y']
        ang = pos['ang']
        label = pos['label']
        pct = pos['pct']
        distance_x = pos['distance_x']
        distance_y = pos['distance_y']
        
        # Check if this label has a manual position override
        if label in manual_positions:
            text_x, text_y = manual_positions[label]
            horizontalalignment = 'left' if text_x > 0 else 'right'
        else:
            # Use automatic positioning for other labels (proj_1, proj_5)
            text_x = distance_x * np.sign(x) * 1.05
            text_y = distance_y * y
            horizontalalignment = 'left' if x > 0 else 'right'
        
        # Connection style
        connectionstyle = f"angle,angleA=0,angleB={ang}"
        
        # Arrow properties
        kw = dict(
            arrowprops=dict(arrowstyle="-", color='gray', lw=1.5, ls='dotted', connectionstyle=connectionstyle),
            bbox=bbox_props,
            zorder=10,
            va="center"
        )
        
        # Create label text with project name and percentage (larger font)
        label_text = f"{label}\n{pct:.1f}%"
        
        ax.annotate(label_text, xy=(x, y), xytext=(text_x, text_y),
                   horizontalalignment=horizontalalignment, fontsize=13, fontweight='bold', **kw)
    
    ax.set_title('Cost Distribution by Project', fontsize=16, fontweight='bold', pad=30)
    
    plt.tight_layout()
    chart_path = f"{ARTIFACTS_DIR}/q1_cost_distribution_pie.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved chart: {chart_path}")
    
    # Chart 3: Daily cost trend for top projects (if date available)
    if 'date' in df.columns and 'cost_usd' in df.columns:
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Prepare data for Seaborn
        top5_projects = top5['project_id'].tolist()
        daily_df = df[df['project_id'].isin(top5_projects)].groupby(['date', 'project_id'])['cost_usd'].sum().reset_index()
        
        # Create line plot with Seaborn
        sns.lineplot(
            data=daily_df,
            x='date',
            y='cost_usd',
            hue='project_id',
            marker='o',
            markersize=5,
            linewidth=2.5,
            palette=COLORS[:5],
            alpha=0.85,
            ax=ax
        )
        
        ax.set_xlabel('Date', fontsize=11, fontweight='bold')
        ax.set_ylabel('Daily Cost (USD)', fontsize=11, fontweight='bold')
        ax.set_title('Daily Cost Trend - Top 5 Projects', fontsize=13, fontweight='bold')
        ax.legend(title='Project', loc='best', fontsize=9)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        chart_path = f"{ARTIFACTS_DIR}/q1_daily_cost_trend.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved chart: {chart_path}")
    
    # ========================================================================
    # STEP 8: PRINT SUMMARY TO CONSOLE
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("TOP 5 PROJECTS SUMMARY")
    print(f"{'='*80}")
    print(f"\n{'Project':<15} {'Cost':>12} {'% Total':>9} {'Driver Type':<20}")
    print("-"*80)
    
    for _, row in top5.iterrows():
        print(f"{row['project_id']:<15} ${row['cost_usd']:>11,.2f} {row['pct_of_total']:>8.1f}% "
              f"{row['driver_type']:<20}")
    
    print("-"*80)
    print(f"{'Total (Top 5)':<15} ${top5['cost_usd'].sum():>11,.2f} "
          f"{top5['pct_of_total'].sum():>8.1f}%")
    print(f"{'Org Total':<15} ${total_cost:>11,.2f}")
    
    # Count by driver type
    driver_counts = top5['driver_type'].value_counts()
    print(f"\nDriver Type Breakdown:")
    for driver, count in driver_counts.items():
        print(f"  • {driver}: {count} project(s)")
    
    # ========================================================================
    # STEP 9: GENERATE EXECUTIVE SUMMARY
    # ========================================================================
    
    # Top project details
    top_project = top5.iloc[0]
    top_project_name = top_project['project_id']
    top_project_cost = top_project['cost_usd']
    top_project_pct = top_project['pct_of_total']
    top_project_driver = top_project['driver_type']
    
    # Top 5 aggregate
    top5_total_cost = top5['cost_usd'].sum()
    top5_total_pct = top5['pct_of_total'].sum()
    
    # Most common driver
    most_common_driver = driver_counts.index[0] if len(driver_counts) > 0 else 'Unknown'
    driver_count = driver_counts.iloc[0] if len(driver_counts) > 0 else 0
    
    # Cost efficiency stats
    if 'cost_per_1k' in top5.columns:
        avg_cost_per_1k = top5['cost_per_1k'].mean()
        cost_efficiency_note = f"Average cost per 1k tokens: ${avg_cost_per_1k:.3f}"
    else:
        cost_efficiency_note = ""
    
    # Premium usage
    if 'pct_premium_tokens' in top5.columns:
        avg_premium = top5['pct_premium_tokens'].mean()
        premium_note = f"Average premium model usage: {avg_premium:.1f}% of tokens"
    else:
        premium_note = ""
    
    # Build executive summary
    exec_summary = f"""
[Q1] SPEND DRIVERS:
  • Top project '{top_project_name}' accounts for ${top_project_cost:,.2f} ({top_project_pct:.1f}% of total) - classified as {top_project_driver}
  • Top 5 projects drive ${top5_total_cost:,.2f} ({top5_total_pct:.1f}% of org spend)"""
    
    if driver_count > 0:
        exec_summary += f"\n  • {driver_count} of top 5 are {most_common_driver}, indicating primary cost pressure"
    
    if cost_efficiency_note:
        exec_summary += f"\n  • {cost_efficiency_note}"
    
    if premium_note:
        exec_summary += f"\n  • {premium_note}"
    
    print(f"\n{'='*80}")
    print("✓ Q1 Analysis Complete")
    print(f"{'='*80}\n")
    
    return exec_summary.strip()


def analyze_q1_project_spend(df):
    """
    Analyze which teams/projects drive spend.
    
    Produces:
    - Top 5 projects by cost chart (bar chart)
    - Summary table with cost, % total, tokens, requests, cost/1k, % premium, cache hit
    
    Args:
        df: Input dataframe with cost and project data
    """
    print("\n" + "="*80)
    print("Q1: WHICH TEAMS/PROJECTS DRIVE SPEND?")
    print("="*80)
    
    required_cols = ['project_id', 'cost_usd']
    missing, available = check_required_columns(df, required_cols)
    
    if missing:
        print(f"❌ Insufficient data for Q1. Missing columns: {missing}")
        return None
    
    # Aggregate by project
    agg_dict = {'cost_usd': 'sum'}
    
    if 'tokens' in df.columns:
        agg_dict['tokens'] = 'sum'
    if 'num_model_requests' in df.columns:
        agg_dict['num_model_requests'] = 'sum'
    if 'is_premium_model' in df.columns:
        df['premium_cost'] = df['cost_usd'] * df['is_premium_model'].astype(int)
        agg_dict['premium_cost'] = 'sum'
    if 'cache_hit_rate' in df.columns:
        # Weighted average cache hit rate
        df['cache_weighted'] = df['cache_hit_rate'] * df['cost_usd']
        agg_dict['cache_weighted'] = 'sum'
    
    project_summary = df.groupby('project_id').agg(agg_dict).reset_index()
    
    # Calculate percentages and derived metrics
    total_cost = project_summary['cost_usd'].sum()
    project_summary['pct_total'] = 100 * project_summary['cost_usd'] / total_cost
    
    if 'tokens' in project_summary.columns:
        project_summary['cost_per_1k'] = safe_divide(
            1000 * project_summary['cost_usd'],
            project_summary['tokens'],
            fill_value=0
        )
    
    if 'premium_cost' in project_summary.columns:
        project_summary['pct_premium'] = 100 * safe_divide(
            project_summary['premium_cost'],
            project_summary['cost_usd'],
            fill_value=0
        )
    
    if 'cache_weighted' in project_summary.columns:
        project_summary['cache_hit_rate'] = safe_divide(
            project_summary['cache_weighted'],
            project_summary['cost_usd'],
            fill_value=0
        )
    
    # Sort by cost and get top 5
    project_summary = project_summary.sort_values('cost_usd', ascending=False)
    top5 = project_summary.head(5).copy()
    
    # Print summary
    print(f"\nTotal spend across {len(project_summary)} projects: ${total_cost:,.2f}")
    print(f"Top 5 projects account for: ${top5['cost_usd'].sum():,.2f} ({100*top5['cost_usd'].sum()/total_cost:.1f}%)")
    
    # Create chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.barh(range(len(top5)), top5['cost_usd'].values)
    ax.set_yticks(range(len(top5)))
    ax.set_yticklabels(top5['project_id'].values)
    ax.set_xlabel('Cost (USD)')
    ax.set_title('Top 5 Projects by Cost')
    ax.invert_yaxis()
    
    # Add value labels on bars
    for i, (cost, pct) in enumerate(zip(top5['cost_usd'].values, top5['pct_total'].values)):
        ax.text(cost, i, f'  ${cost:,.0f} ({pct:.1f}%)', 
                va='center', fontsize=9)
    
    plt.tight_layout()
    chart_path = f"{ARTIFACTS_DIR}/q1_top5_projects_cost.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved chart: {chart_path}")
    
    # Prepare export table
    export_cols = ['project_id', 'cost_usd', 'pct_total']
    if 'tokens' in top5.columns:
        export_cols.append('tokens')
    if 'num_model_requests' in top5.columns:
        export_cols.append('num_model_requests')
    if 'cost_per_1k' in top5.columns:
        export_cols.append('cost_per_1k')
    if 'pct_premium' in top5.columns:
        export_cols.append('pct_premium')
    if 'cache_hit_rate' in top5.columns:
        export_cols.append('cache_hit_rate')
    
    export_table = top5[export_cols].copy()
    
    # Rename columns for clarity
    rename_map = {
        'cost_usd': 'cost_usd',
        'pct_total': 'pct_of_total_cost',
        'cost_per_1k': 'cost_per_1k_tokens',
        'pct_premium': 'pct_premium_model_cost',
        'cache_hit_rate': 'avg_cache_hit_rate'
    }
    export_table.rename(columns=rename_map, inplace=True)
    
    # Save CSV
    csv_path = f"{ARTIFACTS_DIR}/q1_top5_projects_summary.csv"
    export_table.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"✓ Saved summary: {csv_path}")
    
    return export_table


# ============================================================================
# Q2: HOW USAGE PATTERNS CHANGE OVER TIME?
# ============================================================================

def q2_usage_shifts(df):
    """
    Analyze usage patterns over time with shift detection and diagnosis.
    
    Features:
    - Aggregates by day: Requests, Avg Tokens/Req, Cache Hit Rate, Model Token Shares (Top 4)
    - Flags triggers: +30% WoW (requests), ±25% (tokens/req), ±20pp (cache/io), ≥20pp (mix)
    - Saves: 4 small-multiples PNGs and q2_daily_metrics.csv
    - Prints diagnosis bullets per flagged window (Volume/Mix/Prompt/Cache)
    
    Args:
        df: Input dataframe with time-series data
        
    Returns:
        str: Executive summary with key findings
    """
    print("\n" + "="*80)
    print("Q2: HOW USAGE PATTERNS CHANGE OVER TIME?")
    print("="*80)
    
    # Check required columns
    required_cols = ['date']
    missing = [col for col in required_cols if col not in df.columns]
    
    if missing:
        print(f"❌ Insufficient data for Q2. Missing columns: {missing}")
        return "Q2: Insufficient data - missing required columns"
    
    # Work on a copy
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    
    if len(df) == 0:
        return "Q2: No valid date data"
    
    # ========================================================================
    # STEP 1: DAILY AGGREGATION
    # ========================================================================
    
    print("\n  Aggregating daily metrics...")
    
    daily_metrics = {}
    
    # 1. Daily Requests
    if 'num_model_requests' in df.columns:
        daily_metrics['requests'] = df.groupby('date')['num_model_requests'].sum()
    
    # 2. Avg Tokens/Req (weighted by requests)
    if 'tokens_per_req' in df.columns and 'num_model_requests' in df.columns:
        df['tokens_weighted'] = df['tokens_per_req'] * df['num_model_requests']
        daily_tokens_weighted = df.groupby('date')['tokens_weighted'].sum()
        daily_requests = df.groupby('date')['num_model_requests'].sum()
        daily_metrics['avg_tokens_per_req'] = safe_divide(
            daily_tokens_weighted,
            daily_requests,
            fill_value=np.nan
        )
    
    # 3. Cache Hit Rate (weighted by tokens)
    if 'cache_hit_rate' in df.columns and 'tokens' in df.columns:
        df['cache_weighted'] = df['cache_hit_rate'] * df['tokens']
        daily_cache_weighted = df.groupby('date')['cache_weighted'].sum()
        daily_tokens = df.groupby('date')['tokens'].sum()
        daily_metrics['cache_hit_rate'] = safe_divide(
            daily_cache_weighted,
            daily_tokens,
            fill_value=0
        )
    
    # 4. I/O Ratio (weighted by tokens)
    if 'io_ratio' in df.columns and 'tokens' in df.columns:
        df['io_weighted'] = df['io_ratio'] * df['tokens']
        daily_io_weighted = df.groupby('date')['io_weighted'].sum()
        daily_tokens = df.groupby('date')['tokens'].sum()
        daily_metrics['io_ratio'] = safe_divide(
            daily_io_weighted,
            daily_tokens,
            fill_value=np.nan
        )
    
    # 5. Model Token Shares (Top 4 models)
    if 'model' in df.columns and 'tokens' in df.columns:
        # Get top 4 models overall
        top_models = df.groupby('model')['tokens'].sum().nlargest(4).index.tolist()
        
        # Daily tokens by model
        model_daily = df[df['model'].isin(top_models)].groupby(['date', 'model'])['tokens'].sum().unstack(fill_value=0)
        
        # Calculate share
        total_daily_tokens = df.groupby('date')['tokens'].sum()
        for model in top_models:
            if model in model_daily.columns:
                daily_metrics[f'model_share_{model}'] = safe_divide(
                    model_daily[model],
                    total_daily_tokens,
                    fill_value=0
                )
    
    # Create daily dataframe
    daily_df = pd.DataFrame(daily_metrics).sort_index()
    
    print(f"  ✓ Aggregated {len(daily_df)} days of data")
    print(f"  ✓ Date range: {daily_df.index.min().date()} to {daily_df.index.max().date()}")
    
    # ========================================================================
    # STEP 2: DETECT TRIGGERS/FLAGS
    # ========================================================================
    
    print("\n  Detecting usage shifts and triggers...")
    
    flags = []
    
    # Trigger thresholds
    REQUESTS_WOW_THRESHOLD = 0.30  # +30% WoW
    TOKENS_REQ_THRESHOLD = 0.25    # ±25%
    CACHE_IO_THRESHOLD = 0.20      # ±20pp
    MIX_THRESHOLD = 0.20           # ≥20pp
    
    # 1. Requests WoW change (+30%)
    if 'requests' in daily_df.columns:
        requests_wow = daily_df['requests'].pct_change(periods=7)
        
        for date, change in requests_wow.items():
            if pd.notna(change) and change > REQUESTS_WOW_THRESHOLD:
                flags.append({
                    'date': date,
                    'trigger': 'Volume Spike',
                    'metric': 'requests',
                    'change': change,
                    'value': daily_df.loc[date, 'requests'],
                    'diagnosis': 'Volume'
                })
    
    # 2. Tokens/Req shifts - MAJOR changes only (≥100% change OR ≥3000 tokens deviation)
    if 'avg_tokens_per_req' in daily_df.columns:
        baseline_tpr = daily_df['avg_tokens_per_req'].mean()
        
        for date, value in daily_df['avg_tokens_per_req'].items():
            if pd.notna(value) and pd.notna(baseline_tpr):
                pct_change = (value - baseline_tpr) / baseline_tpr
                abs_change = abs(value - baseline_tpr)
                
                # Flag only MAJOR shifts: 100%+ change OR 3000+ tokens deviation
                if abs(pct_change) > 1.0 or abs_change > 3000:
                    flags.append({
                        'date': date,
                        'trigger': 'Prompt Length Shift',
                        'metric': 'avg_tokens_per_req',
                        'change': pct_change,
                        'value': value,
                        'abs_change': abs_change,
                        'diagnosis': 'Prompt'
                    })
    
    # 3. Cache Hit Rate changes (±20pp)
    if 'cache_hit_rate' in daily_df.columns:
        cache_dod = daily_df['cache_hit_rate'].diff()
        
        for date, change in cache_dod.items():
            if pd.notna(change) and abs(change) > CACHE_IO_THRESHOLD:
                flags.append({
                    'date': date,
                    'trigger': 'Cache Pattern Change',
                    'metric': 'cache_hit_rate',
                    'change': change,
                    'value': daily_df.loc[date, 'cache_hit_rate'],
                    'diagnosis': 'Cache'
                })
    
    # 4. I/O Ratio changes (±20pp)
    if 'io_ratio' in daily_df.columns:
        io_dod = daily_df['io_ratio'].diff()
        
        for date, change in io_dod.items():
            if pd.notna(change) and abs(change) > CACHE_IO_THRESHOLD:
                flags.append({
                    'date': date,
                    'trigger': 'I/O Ratio Shift',
                    'metric': 'io_ratio',
                    'change': change,
                    'value': daily_df.loc[date, 'io_ratio'],
                    'diagnosis': 'Prompt'
                })
    
    # 5. Model Mix shifts (≥20pp for any model)
    model_cols = [c for c in daily_df.columns if c.startswith('model_share_')]
    for model_col in model_cols:
        model_dod = daily_df[model_col].diff()
        
        for date, change in model_dod.items():
            if pd.notna(change) and abs(change) > MIX_THRESHOLD:
                model_name = model_col.replace('model_share_', '')
                flags.append({
                    'date': date,
                    'trigger': 'Model Mix Shift',
                    'metric': model_col,
                    'change': change,
                    'value': daily_df.loc[date, model_col],
                    'diagnosis': 'Mix',
                    'model': model_name
                })
    
    print(f"  ✓ Detected {len(flags)} trigger events")
    
    # ========================================================================
    # STEP 3: SAVE CSV
    # ========================================================================
    
    csv_path = f"{ARTIFACTS_DIR}/q2_daily_metrics.csv"
    daily_df.to_csv(csv_path, float_format='%.6f')
    print(f"  ✓ Saved daily metrics: {csv_path}")
    
    # ========================================================================
    # STEP 4: CREATE VISUALIZATIONS (4 SMALL MULTIPLES)
    # ========================================================================
    
    print("\n  Creating visualizations...")
    
    # Chart 1: Daily Requests with flags
    if 'requests' in daily_df.columns:
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # Create line plot with Seaborn
        plot_df = daily_df.reset_index()
        sns.lineplot(
            data=plot_df,
            x='date',
            y='requests',
            marker='o',
            markersize=6,
            linewidth=2.5,
            color=COLOR_PRIMARY,
            label='Daily Requests',
            ax=ax
        )
        
        # Mark volume spike flags
        volume_flags = [f for f in flags if f['diagnosis'] == 'Volume']
        if volume_flags:
            flag_dates = [f['date'] for f in volume_flags]
            flag_values = [f['value'] for f in volume_flags]
            ax.scatter(flag_dates, flag_values, color=COLOR_DANGER, s=120, zorder=5,
                      marker='^', label='Volume Spike', edgecolors='darkred', linewidths=2)
        
        ax.set_xlabel('Date', fontweight='bold')
        ax.set_ylabel('Requests (Millions)', fontweight='bold')
        ax.set_title('Daily Requests (with Volume Spike Flags)', fontsize=13, fontweight='bold')
        
        # Format Y-axis to show values in millions
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.0f}M'))
        
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        chart_path = f"{ARTIFACTS_DIR}/q2_requests.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {chart_path}")
    
    # Chart 2: Avg Tokens/Req with flags
    if 'avg_tokens_per_req' in daily_df.columns:
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # Create line plot with Seaborn
        plot_df = daily_df.reset_index()
        sns.lineplot(
            data=plot_df,
            x='date',
            y='avg_tokens_per_req',
            marker='o',
            markersize=6,
            linewidth=2.5,
            color=COLOR_WARNING,
            label='Avg Tokens/Req',
            ax=ax
        )
        
        # Add baseline
        baseline = daily_df['avg_tokens_per_req'].mean()
        ax.axhline(y=baseline, color='gray', linestyle='--', alpha=0.5, linewidth=2, label=f'Mean: {baseline:.0f}')
        
        # Mark prompt length flags
        prompt_flags = [f for f in flags if f['diagnosis'] == 'Prompt' and 'tokens' in f['metric']]
        if prompt_flags:
            flag_dates = [f['date'] for f in prompt_flags]
            flag_values = [f['value'] for f in prompt_flags]
            ax.scatter(flag_dates, flag_values, color=COLOR_ACCENT, s=120, zorder=5,
                      marker='v', label='Prompt Shift', edgecolors=COLOR_DANGER, linewidths=2)
        
        ax.set_xlabel('Date', fontweight='bold')
        ax.set_ylabel('Tokens per Request', fontweight='bold')
        ax.set_title('Average Tokens per Request (with Prompt Shift Flags)', fontsize=13, fontweight='bold')
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        chart_path = f"{ARTIFACTS_DIR}/q2_tokens_per_req.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {chart_path}")
    
    # Chart 3: Cache Hit Rate with flags
    if 'cache_hit_rate' in daily_df.columns:
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # Create line plot with Seaborn
        plot_df = daily_df.reset_index()
        sns.lineplot(
            data=plot_df,
            x='date',
            y='cache_hit_rate',
            marker='o',
            markersize=6,
            linewidth=2.5,
            color=COLOR_INFO,
            label='Cache Hit Rate',
            ax=ax
        )
        
        # Mark cache flags
        cache_flags = [f for f in flags if f['diagnosis'] == 'Cache']
        if cache_flags:
            flag_dates = [f['date'] for f in cache_flags]
            flag_values = [f['value'] for f in cache_flags]
            ax.scatter(flag_dates, flag_values, color=COLOR_DANGER, s=120, zorder=5,
                      marker='s', label='Cache Shift', edgecolors='darkred', linewidths=2)
        
        ax.set_xlabel('Date', fontweight='bold')
        ax.set_ylabel('Cache Hit Rate (%)', fontweight='bold')
        ax.set_title('Cache Hit Rate (with Cache Shift Flags)\nShifts measured in percentage points (pp)', 
                    fontsize=13, fontweight='bold')
        ax.set_ylim([0, 1])
        
        # Format Y-axis to show percentages
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*100:.0f}%'))
        
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        chart_path = f"{ARTIFACTS_DIR}/q2_cache_rate.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {chart_path}")
    
    # Chart 4: Model Token Shares with flags
    if model_cols:
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # Prepare data for Seaborn (reshape to long format)
        plot_df = daily_df.reset_index()
        model_cols_top4 = model_cols[:4]
        
        # Create a melted dataframe for seaborn
        melt_cols = ['date'] + model_cols_top4
        plot_df_melt = plot_df[melt_cols].melt(id_vars=['date'], var_name='model', value_name='token_share')
        plot_df_melt['model'] = plot_df_melt['model'].str.replace('model_share_', '')
        
        # Create line plot with Seaborn
        sns.lineplot(
            data=plot_df_melt,
            x='date',
            y='token_share',
            hue='model',
            marker='o',
            markersize=5,
            linewidth=2,
            palette=COLORS[:4],
            alpha=0.85,
            ax=ax
        )
        
        # Mark mix flags
        mix_flags = [f for f in flags if f['diagnosis'] == 'Mix']
        if mix_flags:
            flag_dates = [f['date'] for f in mix_flags]
            # Get the value for the specific model
            flag_values = []
            for f in mix_flags:
                if 'model' in f:
                    col_name = f'model_share_{f["model"]}'
                    if col_name in daily_df.columns:
                        flag_values.append(daily_df.loc[f['date'], col_name])
                else:
                    flag_values.append(0.5)  # Default position
            
            ax.scatter(flag_dates, flag_values, color=COLOR_DANGER, s=120, zorder=5,
                      marker='D', label='Mix Shift', edgecolors='darkred', linewidths=2)
        
        ax.set_xlabel('Date', fontweight='bold')
        ax.set_ylabel('Token Share', fontweight='bold')
        ax.set_title('Model Token Share - Top 4 Models (with Mix Shift Flags)', fontsize=13, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.legend(title='Model', fontsize=9)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        chart_path = f"{ARTIFACTS_DIR}/q2_model_mix.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {chart_path}")
    
    # ========================================================================
    # STEP 5: PRINT DIAGNOSIS BULLETS
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("USAGE SHIFT DIAGNOSIS")
    print(f"{'='*80}")
    
    if flags:
        # Group flags by diagnosis type
        diagnosis_groups = {}
        for flag in flags:
            diag = flag['diagnosis']
            if diag not in diagnosis_groups:
                diagnosis_groups[diag] = []
            diagnosis_groups[diag].append(flag)
        
        # Print by diagnosis type
        for diag_type in ['Volume', 'Prompt', 'Cache', 'Mix']:
            if diag_type in diagnosis_groups:
                events = diagnosis_groups[diag_type]
                print(f"\n{diag_type.upper()} Changes ({len(events)} event(s)):")
                
                for event in events[:5]:  # Show top 5 per type
                    date_str = event['date'].strftime('%Y-%m-%d')
                    
                    if diag_type == 'Volume':
                        print(f"  • {date_str}: Requests spike +{event['change']*100:.1f}% WoW "
                              f"(to {event['value']:,.0f} requests)")
                    
                    elif diag_type == 'Prompt':
                        direction = 'increased' if event['change'] > 0 else 'decreased'
                        abs_change = event.get('abs_change', 0)
                        baseline = daily_df['avg_tokens_per_req'].mean() if 'avg_tokens_per_req' in daily_df.columns else 0
                        print(f"  • {date_str}: Avg tokens/req {direction} {abs(event['change'])*100:.1f}% "
                              f"from baseline {baseline:.0f} (to {event['value']:.0f} tokens)")
                    
                    elif diag_type == 'Cache':
                        direction = 'increased' if event['change'] > 0 else 'decreased'
                        print(f"  • {date_str}: Cache hit rate {direction} {abs(event['change'])*100:.1f}pp "
                              f"(to {event['value']*100:.1f}%)")
                    
                    elif diag_type == 'Mix':
                        direction = 'increased' if event['change'] > 0 else 'decreased'
                        model_name = event.get('model', 'unknown')
                        print(f"  • {date_str}: Model '{model_name}' share {direction} {abs(event['change'])*100:.1f}pp "
                              f"(to {event['value']*100:.1f}%)")
                
                if len(events) > 5:
                    print(f"  ... and {len(events) - 5} more")
    else:
        print("\n  No significant usage shifts detected (all metrics stable)")
    
    # ========================================================================
    # STEP 6: GENERATE EXECUTIVE SUMMARY
    # ========================================================================
    
    # Count flags by type
    volume_count = len([f for f in flags if f['diagnosis'] == 'Volume'])
    prompt_count = len([f for f in flags if f['diagnosis'] == 'Prompt'])
    cache_count = len([f for f in flags if f['diagnosis'] == 'Cache'])
    mix_count = len([f for f in flags if f['diagnosis'] == 'Mix'])
    
    # Key metrics
    total_days = len(daily_df)
    avg_requests = daily_df['requests'].mean() if 'requests' in daily_df.columns else 0
    avg_tpr = daily_df['avg_tokens_per_req'].mean() if 'avg_tokens_per_req' in daily_df.columns else 0
    avg_cache = daily_df['cache_hit_rate'].mean() if 'cache_hit_rate' in daily_df.columns else 0
    
    # Build summary
    exec_summary = f"""
[Q2] USAGE TRENDS:
  • Analyzed {total_days} days: {daily_df.index.min().date()} to {daily_df.index.max().date()}"""
    
    if 'requests' in daily_df.columns:
        total_requests = daily_df['requests'].sum()
        exec_summary += f"\n  • Total requests: {total_requests:,.0f} (avg {avg_requests:,.0f}/day)"
    
    # Add trigger summary
    trigger_summary = []
    if volume_count > 0:
        trigger_summary.append(f"{volume_count} volume spike(s)")
    if prompt_count > 0:
        trigger_summary.append(f"{prompt_count} prompt shift(s)")
    if cache_count > 0:
        trigger_summary.append(f"{cache_count} cache change(s)")
    if mix_count > 0:
        trigger_summary.append(f"{mix_count} mix shift(s)")
    
    if trigger_summary:
        exec_summary += f"\n  • Detected shifts: {', '.join(trigger_summary)}"
    else:
        exec_summary += f"\n  • Stable usage patterns (no significant shifts detected)"
    
    if avg_tpr > 0:
        exec_summary += f"\n  • Average tokens/req: {avg_tpr:,.0f}"
    
    if avg_cache > 0:
        exec_summary += f", cache hit rate: {avg_cache:.1%}"
    
    print(f"\n{'='*80}")
    print("✓ Q2 Analysis Complete")
    print(f"{'='*80}\n")
    
    return exec_summary.strip()


def analyze_q2_usage_trends(df):
    """
    Analyze how usage patterns change over time.
    
    Produces:
    - Small multiples showing daily trends for:
      * Requests
      * Avg Tokens/Request
      * Cache Hit Rate
      * Model Token Share (top models)
    
    Args:
        df: Input dataframe with time-series data
    """
    print("\n" + "="*80)
    print("Q2: HOW USAGE PATTERNS CHANGE OVER TIME?")
    print("="*80)
    
    required_cols = ['date']
    missing, available = check_required_columns(df, required_cols)
    
    if missing:
        print(f"❌ Insufficient data for Q2. Missing columns: {missing}")
        return None
    
    # Ensure date is datetime
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    
    # Aggregate by date
    daily_metrics = {}
    
    # Daily requests
    if 'num_model_requests' in df.columns:
        daily_metrics['requests'] = df.groupby('date')['num_model_requests'].sum()
    
    # Daily average tokens per request (weighted)
    if 'tokens_per_req' in df.columns and 'num_model_requests' in df.columns:
        df['tokens_weighted'] = df['tokens_per_req'] * df['num_model_requests']
        daily_tokens_weighted = df.groupby('date')['tokens_weighted'].sum()
        daily_requests = df.groupby('date')['num_model_requests'].sum()
        daily_metrics['avg_tokens_per_req'] = safe_divide(
            daily_tokens_weighted,
            daily_requests,
            fill_value=0
        )
    
    # Daily cache hit rate (weighted by cost)
    if 'cache_hit_rate' in df.columns and 'cost_usd' in df.columns:
        df['cache_weighted'] = df['cache_hit_rate'] * df['cost_usd']
        daily_cache_weighted = df.groupby('date')['cache_weighted'].sum()
        daily_cost = df.groupby('date')['cost_usd'].sum()
        daily_metrics['cache_hit_rate'] = safe_divide(
            daily_cache_weighted,
            daily_cost,
            fill_value=0
        )
    
    # Model token share (top 4 models)
    if 'model' in df.columns and 'tokens' in df.columns:
        # Get top models overall
        top_models = df.groupby('model')['tokens'].sum().nlargest(4).index.tolist()
        
        # Daily tokens by model
        model_daily = df[df['model'].isin(top_models)].groupby(['date', 'model'])['tokens'].sum().unstack(fill_value=0)
        
        # Calculate share
        total_daily_tokens = df.groupby('date')['tokens'].sum()
        for model in top_models:
            if model in model_daily.columns:
                daily_metrics[f'model_share_{model}'] = safe_divide(
                    model_daily[model],
                    total_daily_tokens,
                    fill_value=0
                )
    
    # Create daily summary dataframe
    daily_df = pd.DataFrame(daily_metrics).sort_index()
    
    print(f"\nAnalyzing {len(daily_df)} days of data")
    print(f"Date range: {daily_df.index.min().date()} to {daily_df.index.max().date()}")
    
    # Create small multiples chart
    metrics_to_plot = []
    if 'requests' in daily_df.columns:
        metrics_to_plot.append(('requests', 'Daily Requests', 'Requests'))
    if 'avg_tokens_per_req' in daily_df.columns:
        metrics_to_plot.append(('avg_tokens_per_req', 'Avg Tokens per Request', 'Tokens/Req'))
    if 'cache_hit_rate' in daily_df.columns:
        metrics_to_plot.append(('cache_hit_rate', 'Cache Hit Rate', 'Rate'))
    
    # Add top model if available
    model_cols = [c for c in daily_df.columns if c.startswith('model_share_')]
    if model_cols:
        metrics_to_plot.append((model_cols[:4], 'Model Token Share (Top Models)', 'Share'))
    
    n_plots = len(metrics_to_plot)
    
    if n_plots == 0:
        print("❌ Insufficient metrics for time series analysis")
        return None
    
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3*n_plots))
    
    if n_plots == 1:
        axes = [axes]
    
    for idx, (metric, title, ylabel) in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        if isinstance(metric, list):
            # Multiple series (model shares)
            for m in metric:
                model_name = m.replace('model_share_', '')
                ax.plot(daily_df.index, daily_df[m], marker='o', markersize=4, 
                       label=model_name, linewidth=1.5)
            ax.legend(fontsize=8)
        else:
            # Single series
            ax.plot(daily_df.index, daily_df[metric], marker='o', markersize=4, 
                   linewidth=1.5, color='steelblue')
        
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    axes[-1].set_xlabel('Date')
    
    plt.tight_layout()
    chart_path = f"{ARTIFACTS_DIR}/q2_usage_trends.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved chart: {chart_path}")
    
    # Save daily metrics CSV
    csv_path = f"{ARTIFACTS_DIR}/q2_daily_metrics.csv"
    daily_df.to_csv(csv_path, float_format='%.6f')
    print(f"✓ Saved daily metrics: {csv_path}")
    
    return daily_df


# ============================================================================
# Q3: ARE COSTS ON-BUDGET OR AT RISK?
# ============================================================================

def q3_budget(df, monthly_budget=MONTHLY_BUDGET):
    """
    Analyze budget tracking with burn-up, projections, and risk contributors.
    
    Features:
    - Burn-up chart: cumulative cost by day vs linear budget line
    - Run rate: avg last 7 days
    - Projected EOM & Pacing index
    - Risk contributors: projects with projected_EOM/budget > 1 (if per-project budgets available)
    - Saves: q3_burnup.png, q3_budget_table.csv
    - Prints: 2-4 exec bullets
    
    Args:
        df: Input dataframe with cost and date data
        monthly_budget: Monthly budget in USD (default: MONTHLY_BUDGET constant)
        
    Returns:
        str: Executive summary with key findings
    """
    print("\n" + "="*80)
    print("Q3: ARE COSTS ON-BUDGET OR AT RISK?")
    print("="*80)
    
    # Check required columns
    required_cols = ['date', 'cost_usd']
    missing = [col for col in required_cols if col not in df.columns]
    
    if missing:
        print(f"❌ Insufficient data for Q3. Missing columns: {missing}")
        return "Q3: Insufficient data - missing required columns"
    
    # Work on a copy
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    
    if len(df) == 0:
        return "Q3: No valid date data"
    
    print(f"\n  Monthly budget: ${monthly_budget:,.2f}")
    
    # ========================================================================
    # STEP 1: DAILY AGGREGATION & BURN-UP
    # ========================================================================
    
    print("\n  Calculating burn-up metrics...")
    
    # Daily cost aggregation
    daily_cost = df.groupby('date')['cost_usd'].sum().sort_index()
    
    # Cumulative cost (burn-up)
    cumulative_cost = daily_cost.cumsum()
    
    # Date range
    start_date = daily_cost.index.min()
    end_date = daily_cost.index.max()
    days_elapsed = (end_date - start_date).days + 1
    
    # Assume month has 30 days (or use actual days in month)
    days_in_month = DAYS_IN_MONTH
    days_remaining = max(0, days_in_month - days_elapsed)
    
    # MTD cost
    mtd_cost = cumulative_cost.iloc[-1]
    
    # Run rate (last 7 days average)
    last_7_days = daily_cost.tail(7)
    run_rate_daily = last_7_days.mean()
    
    # Projected EOM
    projected_eom = mtd_cost + (run_rate_daily * days_remaining)
    
    # Pacing index
    pacing_index = safe_divide(mtd_cost, monthly_budget, 0) / safe_divide(days_elapsed, days_in_month, 1)
    
    # Budget status
    budget_variance = projected_eom - monthly_budget
    budget_variance_pct = safe_divide(budget_variance, monthly_budget, 0) * 100
    
    print(f"  ✓ MTD cost: ${mtd_cost:,.2f} ({mtd_cost/monthly_budget*100:.1f}% of budget)")
    print(f"  ✓ Run rate: ${run_rate_daily:,.2f}/day (7-day avg)")
    print(f"  ✓ Projected EOM: ${projected_eom:,.2f}")
    print(f"  ✓ Pacing index: {pacing_index:.2f}x")
    
    # ========================================================================
    # STEP 2: RISK CONTRIBUTORS (IF PROJECT BUDGETS AVAILABLE)
    # ========================================================================
    
    risk_contributors = []
    has_project_budgets = False
    
    # Check if project budgets are available (look for project_budget column)
    if 'project_id' in df.columns and 'project_budget' in df.columns:
        print("\n  Analyzing project-level budget risk...")
        has_project_budgets = True
        
        # Aggregate by project
        project_summary = df.groupby('project_id').agg({
            'cost_usd': 'sum',
            'project_budget': 'first'  # Assume consistent per project
        }).reset_index()
        
        # Calculate project-level metrics
        for _, row in project_summary.iterrows():
            project_id = row['project_id']
            project_mtd = row['cost_usd']
            project_budget = row['project_budget']
            
            if pd.notna(project_budget) and project_budget > 0:
                # Project run rate (proportional to org run rate)
                project_daily_avg = project_mtd / days_elapsed
                project_projected_eom = project_mtd + (project_daily_avg * days_remaining)
                
                # Risk ratio
                risk_ratio = project_projected_eom / project_budget
                
                if risk_ratio > 1.0:  # Over budget
                    risk_contributors.append({
                        'project_id': project_id,
                        'mtd_cost': project_mtd,
                        'project_budget': project_budget,
                        'projected_eom': project_projected_eom,
                        'risk_ratio': risk_ratio,
                        'variance': project_projected_eom - project_budget,
                        'variance_pct': (risk_ratio - 1) * 100
                    })
        
        # Sort by risk ratio
        risk_contributors = sorted(risk_contributors, key=lambda x: x['risk_ratio'], reverse=True)
        
        print(f"  ✓ Identified {len(risk_contributors)} at-risk projects (projected > budget)")
    
    elif 'project_id' in df.columns:
        print("\n  ℹ️  Per-project budgets not available (no 'project_budget' column)")
        print("  ℹ️  Skipping project-level risk analysis (gracefully)")
        
        # Alternative: Show top spenders even without budgets
        project_summary = df.groupby('project_id')['cost_usd'].sum().sort_values(ascending=False).head(5)
        print(f"\n  Top 5 spending projects:")
        for proj, cost in project_summary.items():
            pct = (cost / mtd_cost) * 100
            print(f"    • {proj}: ${cost:,.2f} ({pct:.1f}% of MTD)")
    
    # ========================================================================
    # STEP 3: CREATE BURN-UP CHART
    # ========================================================================
    
    print("\n  Creating burn-up visualization...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Actual cumulative cost
    ax.plot(cumulative_cost.index, cumulative_cost.values,
           marker='o', markersize=5, linewidth=2.5,
           label='Actual Cumulative Cost', color=COLOR_SECONDARY, zorder=3)
    
    # Linear budget line (full month)
    budget_dates = pd.date_range(start_date, start_date + pd.Timedelta(days=days_in_month-1), freq='D')
    budget_line = np.linspace(0, monthly_budget, days_in_month)
    
    ax.plot(budget_dates, budget_line,
           linestyle='--', linewidth=2, color=COLOR_PRIMARY,
           label='Linear Budget', alpha=0.7, zorder=2)
    
    # Projected line (from current to EOM)
    if days_remaining > 0:
        projection_dates = pd.date_range(end_date, end_date + pd.Timedelta(days=days_remaining), freq='D')
        projection_values = np.linspace(mtd_cost, projected_eom, len(projection_dates))
        ax.plot(projection_dates, projection_values,
               linestyle=':', linewidth=2.5, color=COLOR_WARNING,
               label='Projected', alpha=0.8, zorder=2)
    
    # Budget limit line
    ax.axhline(y=monthly_budget, color=COLOR_DANGER, linestyle='-',
              linewidth=2, alpha=0.6, label='Budget Limit', zorder=1)
    
    # Add annotation for current status
    ax.scatter([end_date], [mtd_cost], color='darkblue', s=200, zorder=5,
              marker='o', edgecolors='black', linewidths=2)
    
    # Annotation box
    status_color = 'salmon' if pacing_index > 1.1 else 'lightgreen' if pacing_index < 0.9 else 'lightyellow'
    annotation_text = (
        f"Current Status (Day {days_elapsed}):\n"
        f"MTD: ${mtd_cost:,.0f}\n"
        f"Projected EOM: ${projected_eom:,.0f}\n"
        f"Pacing: {pacing_index:.2f}x\n"
        f"Variance: ${budget_variance:+,.0f} ({budget_variance_pct:+.1f}%)"
    )
    
    ax.annotate(annotation_text,
               xy=(end_date, mtd_cost),
               xytext=(20, 30), textcoords='offset points',
               bbox=dict(boxstyle='round,pad=0.8', facecolor=status_color, alpha=0.8, edgecolor='black', linewidth=1.5),
               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', lw=2),
               fontsize=10, fontweight='bold')
    
    # Styling
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Cost (USD)', fontsize=12, fontweight='bold')
    ax.set_title('Budget Burn-up: Actual vs Plan', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(rotation=45)
    
    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.tight_layout()
    chart_path = f"{ARTIFACTS_DIR}/q3_burnup.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {chart_path}")
    
    # ========================================================================
    # ADDITIONAL Q3 CHARTS
    # ========================================================================
    
    # Chart 2: Daily Cost Bar Chart
    fig, ax = plt.subplots(figsize=(14, 6))
    
    daily_cost_series = daily_cost.sort_index()
    
    # Prepare data for Seaborn - format dates as strings
    plot_df = pd.DataFrame({
        'date': [d.strftime('%Y-%m-%d') for d in daily_cost_series.index],  # Format as date string only
        'cost': daily_cost_series.values
    })
    plot_df['status'] = plot_df['cost'].apply(
        lambda x: 'Under Budget' if x <= (monthly_budget / days_in_month) else 'Over Budget'
    )
    
    # Create bar plot with Seaborn
    sns.barplot(
        data=plot_df,
        x='date',
        y='cost',
        hue='status',
        palette={'Under Budget': COLOR_PRIMARY, 'Over Budget': COLOR_DANGER},
        alpha=0.85,
        ax=ax,
        dodge=False
    )
    
    # Add average line
    avg_daily_budget = monthly_budget / days_in_month
    ax.axhline(y=avg_daily_budget, color=COLOR_WARNING, linestyle='--', 
              linewidth=2, label=f'Daily Budget Target: ${avg_daily_budget:,.0f}', alpha=0.7)
    
    # Add run-rate line
    ax.axhline(y=run_rate_daily, color=COLOR_ACCENT, linestyle='-', 
              linewidth=2, label=f'Average Rate: ${run_rate_daily:,.0f}', alpha=0.7)
    
    ax.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax.set_ylabel('Daily Cost (USD)', fontsize=11, fontweight='bold')
    ax.set_title('Daily Spending Pattern', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    
    # Format x-axis: vertical dates, closer to graph
    plt.xticks(rotation=90, ha='center')
    ax.tick_params(axis='x', pad=2)  # Reduce padding between labels and axis
    
    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.tight_layout()
    chart_path = f"{ARTIFACTS_DIR}/q3_daily_costs.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {chart_path}")
    
    # Chart 3: Pacing Gauge (Visual Indicator)
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': 'polar'})
    
    # Gauge visualization using polar plot
    theta = np.linspace(0, np.pi, 100)
    
    # Background zones
    r = np.ones(100)
    ax.fill_between(theta[0:33], 0, r[0:33], color=COLOR_PRIMARY, alpha=0.3, label='Under Budget')
    ax.fill_between(theta[33:67], 0, r[33:67], color=COLOR_WARNING, alpha=0.3, label='On Track')
    ax.fill_between(theta[67:100], 0, r[67:100], color=COLOR_DANGER, alpha=0.3, label='Over Budget')
    
    # Needle position based on pacing index
    # Map pacing index: 0.5x -> 0°, 1.0x -> 90°, 2.0x -> 180°
    if pacing_index <= 0.5:
        needle_angle = 0
    elif pacing_index <= 2.0:
        needle_angle = (pacing_index - 0.5) / 1.5 * np.pi
    else:
        needle_angle = np.pi
    
    ax.plot([needle_angle, needle_angle], [0, 0.9], color='black', linewidth=4, marker='o', 
           markersize=10, markerfacecolor='black')
    
    # Remove radial ticks and labels
    ax.set_yticks([])
    ax.set_xticks([0, np.pi/3, 2*np.pi/3, np.pi])
    ax.set_xticklabels(['0.5x\nUnder', '1.0x\nOn Track', '1.5x\nRisk', '2.0x+\nOver'], fontsize=10)
    ax.set_theta_zero_location('W')
    ax.set_theta_direction(1)
    ax.set_ylim(0, 1)
    
    # Title with pacing value
    ax.set_title(f'Budget Pacing Gauge\nCurrent: {pacing_index:.2f}x', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    chart_path = f"{ARTIFACTS_DIR}/q3_pacing_gauge.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {chart_path}")
    
    # ========================================================================
    # STEP 4: SAVE BUDGET TABLE
    # ========================================================================
    
    print("\n  Saving budget summary table...")
    
    # Main budget metrics
    budget_table = pd.DataFrame({
        'metric': [
            'monthly_budget',
            'days_elapsed',
            'days_remaining',
            'days_in_month',
            'mtd_cost',
            'run_rate_daily',
            'projected_eom',
            'pacing_index',
            'budget_variance',
            'budget_variance_pct'
        ],
        'value': [
            monthly_budget,
            days_elapsed,
            days_remaining,
            days_in_month,
            mtd_cost,
            run_rate_daily,
            projected_eom,
            pacing_index,
            budget_variance,
            budget_variance_pct
        ]
    })
    
    csv_path = f"{ARTIFACTS_DIR}/q3_budget_table.csv"
    budget_table.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"  ✓ Saved: {csv_path}")
    
    # Save risk contributors if available
    if risk_contributors:
        risk_df = pd.DataFrame(risk_contributors)
        risk_csv_path = f"{ARTIFACTS_DIR}/q3_risk_contributors.csv"
        risk_df.to_csv(risk_csv_path, index=False, float_format='%.4f')
        print(f"  ✓ Saved risk contributors: {risk_csv_path}")
    
    # ========================================================================
    # STEP 5: PRINT SUMMARY
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("BUDGET SUMMARY")
    print(f"{'='*80}")
    
    print(f"\nOrganization Budget:")
    print(f"  Monthly Budget:    ${monthly_budget:>12,.2f}")
    print(f"  MTD Cost:          ${mtd_cost:>12,.2f}  ({mtd_cost/monthly_budget*100:>5.1f}%)")
    print(f"  Projected EOM:     ${projected_eom:>12,.2f}  ({projected_eom/monthly_budget*100:>5.1f}%)")
    print(f"  Variance:          ${budget_variance:>12,.2f}  ({budget_variance_pct:>+5.1f}%)")
    print(f"  Pacing Index:      {pacing_index:>12.2f}x", end='')
    
    if pacing_index > 1.1:
        print("  ⚠️  OVER-PACING")
    elif pacing_index < 0.9:
        print("  ℹ️  Under-pacing")
    else:
        print("  ✓  On track")
    
    print(f"\nRun Rate Analysis:")
    print(f"  Last 7 days avg:   ${run_rate_daily:>12,.2f}/day")
    print(f"  Days remaining:    {days_remaining:>12} days")
    print(f"  Burn to EOM:       ${run_rate_daily * days_remaining:>12,.2f}")
    
    # Risk contributors summary
    if risk_contributors:
        print(f"\nAt-Risk Projects ({len(risk_contributors)}):")
        for i, risk in enumerate(risk_contributors[:5], 1):
            print(f"  {i}. {risk['project_id']:<15} "
                  f"Proj: ${risk['projected_eom']:>10,.0f} / Budget: ${risk['project_budget']:>10,.0f} "
                  f"({risk['risk_ratio']:.2f}x)")
        if len(risk_contributors) > 5:
            print(f"  ... and {len(risk_contributors) - 5} more")
    
    # ========================================================================
    # STEP 6: GENERATE EXECUTIVE SUMMARY
    # ========================================================================
    
    # Determine status
    if pacing_index > 1.1:
        status = "⚠️ OVER-BUDGET RISK"
    elif pacing_index < 0.9:
        status = "under-pacing"
    else:
        status = "on track"
    
    # Build summary
    exec_summary = f"""
[Q3] BUDGET STATUS:
  • MTD spend: ${mtd_cost:,.2f} ({mtd_cost/monthly_budget*100:.1f}% of ${monthly_budget:,.0f} budget)
  • Projected EOM: ${projected_eom:,.2f} (variance: ${budget_variance:+,.0f}, {budget_variance_pct:+.1f}%)"""
    
    exec_summary += f"\n  • Pacing index: {pacing_index:.2f}x - {status}"
    
    if risk_contributors:
        exec_summary += f"\n  • {len(risk_contributors)} project(s) at risk of exceeding individual budgets"
    else:
        exec_summary += f"\n  • Run rate: ${run_rate_daily:,.2f}/day over last 7 days"
    
    print(f"\n{'='*80}")
    print("✓ Q3 Analysis Complete")
    print(f"{'='*80}\n")
    
    return exec_summary.strip()


def analyze_q3_budget_tracking(df, monthly_budget=MONTHLY_BUDGET):
    """
    Analyze budget tracking and risk.
    
    Produces:
    - Burn-up chart vs budget
    - Projected EOM cost
    - Pacing index
    
    Args:
        df: Input dataframe with cost and date data
        monthly_budget: Monthly budget in USD
    """
    print("\n" + "="*80)
    print("Q3: ARE COSTS ON-BUDGET OR AT RISK?")
    print("="*80)
    
    required_cols = ['date', 'cost_usd']
    missing, available = check_required_columns(df, required_cols)
    
    if missing:
        print(f"❌ Insufficient data for Q3. Missing columns: {missing}")
        return None
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    
    # Daily cost aggregation
    daily_cost = df.groupby('date')['cost_usd'].sum().sort_index()
    
    # Cumulative cost
    cumulative_cost = daily_cost.cumsum()
    
    # Get date range
    start_date = daily_cost.index.min()
    end_date = daily_cost.index.max()
    days_elapsed = (end_date - start_date).days + 1
    
    # Assume month has 30 days for calculation
    days_in_month = DAYS_IN_MONTH
    
    # MTD cost
    mtd_cost = cumulative_cost.iloc[-1]
    
    # Run rate (last 7 days average)
    last_7_days = daily_cost.tail(7)
    run_rate_daily = last_7_days.mean()
    days_remaining = max(0, days_in_month - days_elapsed)
    
    # Projected EOM
    projected_eom = mtd_cost + (run_rate_daily * days_remaining)
    
    # Pacing index
    budget_pacing = safe_divide(mtd_cost, monthly_budget, 0) / safe_divide(days_elapsed, days_in_month, 1)
    
    print(f"\nBudget Analysis:")
    print(f"  Date range: {start_date.date()} to {end_date.date()} ({days_elapsed} days)")
    print(f"  Monthly budget: ${monthly_budget:,.2f}")
    print(f"  MTD cost: ${mtd_cost:,.2f} ({100*mtd_cost/monthly_budget:.1f}% of budget)")
    print(f"  Run rate (7-day avg): ${run_rate_daily:,.2f}/day")
    print(f"  Projected EOM: ${projected_eom:,.2f}")
    print(f"  Pacing index: {budget_pacing:.2f}x {'⚠️ OVER-PACING' if budget_pacing > 1.1 else '✓ On track' if budget_pacing > 0.9 else '⚠️ Under-pacing'}")
    
    # Create burn-up chart
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Actual cumulative cost
    ax.plot(cumulative_cost.index, cumulative_cost.values, 
           marker='o', markersize=4, linewidth=2, 
           label='Actual Cumulative Cost', color='steelblue')
    
    # Linear budget line
    date_range = pd.date_range(start_date, start_date + pd.Timedelta(days=days_in_month-1), freq='D')
    budget_line = np.linspace(0, monthly_budget, days_in_month)
    
    # Only plot budget line up to current date + projection
    ax.plot(date_range[:days_elapsed], budget_line[:days_elapsed], 
           linestyle='--', linewidth=2, color='green', 
           label='Linear Budget', alpha=0.7)
    
    # Projected line
    if days_remaining > 0:
        projection_dates = pd.date_range(end_date, end_date + pd.Timedelta(days=days_remaining), freq='D')
        projection_values = np.linspace(mtd_cost, projected_eom, len(projection_dates))
        ax.plot(projection_dates, projection_values, 
               linestyle=':', linewidth=2, color='orange', 
               label='Projected', alpha=0.7)
    
    # Budget limit line
    ax.axhline(y=monthly_budget, color='red', linestyle='-', 
              linewidth=1.5, alpha=0.5, label='Budget Limit')
    
    # Add annotation for projected EOM
    ax.text(end_date, projected_eom, 
           f'  Projected EOM: ${projected_eom:,.0f}\n  Pacing: {budget_pacing:.2f}x', 
           fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Cost (USD)')
    ax.set_title('Budget Burn-up vs Linear Budget')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    chart_path = f"{ARTIFACTS_DIR}/q3_budget_burnup.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved chart: {chart_path}")
    
    # Save budget summary
    budget_summary = pd.DataFrame({
        'metric': [
            'monthly_budget',
            'days_elapsed',
            'days_remaining',
            'mtd_cost',
            'run_rate_daily',
            'projected_eom',
            'pacing_index',
            'budget_remaining',
            'projected_variance'
        ],
        'value': [
            monthly_budget,
            days_elapsed,
            days_remaining,
            mtd_cost,
            run_rate_daily,
            projected_eom,
            budget_pacing,
            monthly_budget - mtd_cost,
            projected_eom - monthly_budget
        ]
    })
    
    csv_path = f"{ARTIFACTS_DIR}/q3_budget_summary.csv"
    budget_summary.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"✓ Saved budget summary: {csv_path}")
    
    return budget_summary


# ============================================================================
# Q4: WHERE ANOMALIES, SPIKES, OR INEFFICIENCIES HAPPEN?
# ============================================================================

def q4_anomalies(df):
    """
    Detect anomalies and inefficiencies with structured indices.
    
    Features:
    - Spike Index per day = avg z-score of requests and tokens (14-day baseline)
    - Inefficiency Index per project = 0.5*(cost/1k÷org_median) + 0.3*(premium_share) + 0.2*(1-cache_rate)
    - Flags days with Spike Index > 3
    - Flags projects with Inefficiency Index > 1.3
    - Saves: q4_anomalies.png (panel), q4_flags.csv
    - Prints: concise bullets
    
    Args:
        df: Input dataframe with usage and cost data
        
    Returns:
        str: Executive summary with key findings
    """
    print("\n" + "="*80)
    print("Q4: WHERE ANOMALIES, SPIKES, OR INEFFICIENCIES HAPPEN?")
    print("="*80)
    
    # Check required columns
    required_cols = ['date']
    missing = [col for col in required_cols if col not in df.columns]
    
    if missing:
        print(f"❌ Insufficient data for Q4. Missing columns: {missing}")
        return "Q4: Insufficient data - missing required columns"
    
    # Work on a copy
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    
    if len(df) == 0:
        return "Q4: No valid date data"
    
    # ========================================================================
    # STEP 1: SPIKE INDEX (PER DAY)
    # ========================================================================
    
    print("\n  Computing Spike Index (daily)...")
    
    spike_flags = []
    daily_spike_index = None
    
    if 'num_model_requests' in df.columns and 'tokens' in df.columns:
        # Daily aggregation
        daily_requests = df.groupby('date')['num_model_requests'].sum().sort_index()
        daily_tokens = df.groupby('date')['tokens'].sum().sort_index()
        
        # Calculate z-scores (14-day rolling baseline)
        baseline_window = BASELINE_DAYS
        
        # Requests z-score
        requests_rolling_mean = daily_requests.rolling(window=baseline_window, min_periods=1).mean()
        requests_rolling_std = daily_requests.rolling(window=baseline_window, min_periods=1).std()
        requests_z_values = safe_divide(
            daily_requests - requests_rolling_mean,
            requests_rolling_std,
            fill_value=0
        )
        # Convert back to Series if needed
        if isinstance(requests_z_values, np.ndarray):
            requests_z = pd.Series(requests_z_values, index=daily_requests.index)
        else:
            requests_z = requests_z_values
        
        # Tokens z-score
        tokens_rolling_mean = daily_tokens.rolling(window=baseline_window, min_periods=1).mean()
        tokens_rolling_std = daily_tokens.rolling(window=baseline_window, min_periods=1).std()
        tokens_z_values = safe_divide(
            daily_tokens - tokens_rolling_mean,
            tokens_rolling_std,
            fill_value=0
        )
        # Convert back to Series if needed
        if isinstance(tokens_z_values, np.ndarray):
            tokens_z = pd.Series(tokens_z_values, index=daily_tokens.index)
        else:
            tokens_z = tokens_z_values
        
        # Spike Index = average of z-scores
        spike_index = (requests_z + tokens_z) / 2
        
        # Store for visualization
        daily_spike_index = pd.DataFrame({
            'date': spike_index.index,
            'spike_index': spike_index.values,
            'requests_z': requests_z.values,
            'tokens_z': tokens_z.values,
            'requests': daily_requests.values,
            'tokens': daily_tokens.values
        })
        
        # Flag days with Spike Index > 3
        flagged_days = spike_index[spike_index > 3]
        
        for date, index_val in flagged_days.items():
            spike_flags.append({
                'type': 'Spike',
                'date': date,
                'entity': 'Organization',
                'spike_index': index_val,
                'requests_z': requests_z.loc[date],
                'tokens_z': tokens_z.loc[date],
                'requests': daily_requests.loc[date],
                'tokens': daily_tokens.loc[date]
            })
        
        print(f"  ✓ Calculated Spike Index for {len(spike_index)} days")
        print(f"  ✓ Flagged {len(spike_flags)} days with Spike Index > 3")
    
    else:
        print("  ⚠️  Missing columns for Spike Index (need: num_model_requests, tokens)")
    
    # ========================================================================
    # STEP 2: INEFFICIENCY INDEX (PER PROJECT)
    # ========================================================================
    
    print("\n  Computing Inefficiency Index (by project)...")
    
    inefficiency_flags = []
    project_inefficiency = None
    
    if 'project_id' in df.columns:
        # Aggregate by project
        agg_dict = {'cost_usd': 'sum'}
        
        if 'tokens' in df.columns:
            agg_dict['tokens'] = 'sum'
        if 'is_premium_model' in df.columns and 'tokens' in df.columns:
            df['premium_tokens'] = df['tokens'] * df['is_premium_model'].astype(int)
            agg_dict['premium_tokens'] = 'sum'
        if 'cache_hit_rate' in df.columns and 'tokens' in df.columns:
            df['cache_weighted'] = df['cache_hit_rate'] * df['tokens']
            agg_dict['cache_weighted'] = 'sum'
        
        project_summary = df.groupby('project_id').agg(agg_dict).reset_index()
        
        # Calculate components
        
        # 1. Cost per 1k ratio (vs org median)
        if 'tokens' in project_summary.columns:
            project_summary['cost_per_1k'] = safe_divide(
                1000 * project_summary['cost_usd'],
                project_summary['tokens'],
                fill_value=np.nan
            )
            org_median_cost = project_summary['cost_per_1k'].median()
            project_summary['cost_ratio'] = safe_divide(
                project_summary['cost_per_1k'],
                org_median_cost,
                fill_value=1.0
            )
        else:
            project_summary['cost_ratio'] = 1.0
        
        # 2. Premium share
        if 'premium_tokens' in project_summary.columns and 'tokens' in project_summary.columns:
            project_summary['premium_share'] = safe_divide(
                project_summary['premium_tokens'],
                project_summary['tokens'],
                fill_value=0
            )
        else:
            project_summary['premium_share'] = 0
        
        # 3. Cache hit rate (weighted average)
        if 'cache_weighted' in project_summary.columns and 'tokens' in project_summary.columns:
            project_summary['cache_hit_rate'] = safe_divide(
                project_summary['cache_weighted'],
                project_summary['tokens'],
                fill_value=0
            )
        else:
            project_summary['cache_hit_rate'] = 0
        
        # Calculate Inefficiency Index
        # 0.5 * cost_ratio + 0.3 * premium_share + 0.2 * (1 - cache_hit_rate)
        project_summary['inefficiency_index'] = (
            0.5 * project_summary['cost_ratio'] +
            0.3 * project_summary['premium_share'] +
            0.2 * (1 - project_summary['cache_hit_rate'])
        )
        
        # Store for visualization
        project_inefficiency = project_summary.copy()
        
        # Flag projects with Inefficiency Index > 1.3
        flagged_projects = project_summary[project_summary['inefficiency_index'] > 1.3].sort_values(
            'inefficiency_index', ascending=False
        )
        
        for _, row in flagged_projects.iterrows():
            inefficiency_flags.append({
                'type': 'Inefficiency',
                'date': None,
                'entity': row['project_id'],
                'inefficiency_index': row['inefficiency_index'],
                'cost_ratio': row['cost_ratio'],
                'premium_share': row['premium_share'],
                'cache_hit_rate': row['cache_hit_rate'],
                'cost_usd': row['cost_usd']
            })
        
        print(f"  ✓ Calculated Inefficiency Index for {len(project_summary)} projects")
        print(f"  ✓ Flagged {len(inefficiency_flags)} projects with Inefficiency Index > 1.3")
    
    else:
        print("  ⚠️  Missing project_id column for Inefficiency Index")
    
    # ========================================================================
    # STEP 3: SAVE FLAGS CSV
    # ========================================================================
    
    print("\n  Saving flags...")
    
    # Combine all flags
    all_flags = spike_flags + inefficiency_flags
    
    if all_flags:
        flags_df = pd.DataFrame(all_flags)
        csv_path = f"{ARTIFACTS_DIR}/q4_flags.csv"
        flags_df.to_csv(csv_path, index=False, float_format='%.4f')
        print(f"  ✓ Saved: {csv_path} ({len(all_flags)} flags)")
    else:
        print("  ℹ️  No flags detected (all metrics within thresholds)")
    
    # ========================================================================
    # STEP 4: CREATE PANEL VISUALIZATION (Spike Index Only)
    # ========================================================================
    
    print("\n  Creating anomaly visualization (Spike Index)...")
    
    # Only show Spike Index (Inefficiency Index shown in separate breakdown chart)
    if daily_spike_index is not None:
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        
        # Plot spike index with Seaborn
        sns.lineplot(
            data=daily_spike_index,
            x='date',
            y='spike_index',
            marker='o',
            markersize=6,
            linewidth=2.5,
            color=COLOR_SECONDARY,
            label='Token Spike Index',
            ax=ax
        )
        
        # Threshold line
        ax.axhline(y=3, color=COLOR_DANGER, linestyle='--', linewidth=2,
                  label='Threshold (3.0)', alpha=0.7, zorder=1)
        
        # Mark flagged days
        if spike_flags:
            flag_dates = [f['date'] for f in spike_flags]
            flag_values = [f['spike_index'] for f in spike_flags]
            ax.scatter(flag_dates, flag_values, color=COLOR_ACCENT, s=150, zorder=5,
                      marker='X', edgecolors=COLOR_DANGER, linewidths=2,
                      label='Flagged Token Spikes')
        
        ax.set_xlabel('Date', fontsize=11, fontweight='bold')
        ax.set_ylabel('Token Spike Index (avg z-score)', fontsize=11, fontweight='bold')
        ax.set_title('Daily Token Spike Index (Requests + Tokens)', fontsize=13, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        chart_path = f"{ARTIFACTS_DIR}/q4_anomalies.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {chart_path}")
    else:
        print("  ⚠️  Insufficient data for Spike Index visualization")
    
    # ========================================================================
    # ADDITIONAL Q4 CHART: Inefficiency Components Breakdown
    # ========================================================================
    
    if project_inefficiency is not None and len(project_inefficiency) > 0:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Get top 10 projects
        top_10 = project_inefficiency.nlargest(10, 'inefficiency_index')
        
        # Prepare data for stacked bar
        projects = top_10['project_id'].values
        cost_component = 0.5 * top_10['cost_ratio'].values
        premium_component = 0.3 * top_10['premium_share'].values
        cache_component = 0.2 * (1 - top_10['cache_hit_rate'].values)
        
        x = np.arange(len(projects))
        width = 0.6
        
        # Create stacked bars
        p1 = ax.bar(x, cost_component, width, label='Cost Ratio (50%)', 
                   color=COLOR_DANGER, alpha=0.8)
        p2 = ax.bar(x, premium_component, width, bottom=cost_component,
                   label='Premium Usage (30%)', color=COLOR_WARNING, alpha=0.8)
        p3 = ax.bar(x, cache_component, width, 
                   bottom=cost_component + premium_component,
                   label='Cache Miss (20%)', color=COLOR_INFO, alpha=0.8)
        
        # Add threshold lines
        ax.axhline(y=1.3, color=COLOR_ACCENT, linestyle='--', linewidth=2,
                  label='Inefficiency Threshold (1.3)', alpha=0.8)
        ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1.5,
                  label='Baseline (1.0)', alpha=0.6)
        
        # Formatting
        ax.set_ylabel('Inefficiency Index', fontsize=12, fontweight='bold')
        ax.set_xlabel('Project', fontsize=12, fontweight='bold')
        ax.set_title('Inefficiency Index Breakdown by Component', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(projects, rotation=45, ha='right')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add total value labels on top
        for i, (cost, prem, cache) in enumerate(zip(cost_component, premium_component, cache_component)):
            total = cost + prem + cache
            ax.text(i, total, f'{total:.2f}', ha='center', va='bottom', 
                   fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        chart_path = f"{ARTIFACTS_DIR}/q4_inefficiency_breakdown.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {chart_path}")
    
    # ========================================================================
    # STEP 5: PRINT CONCISE BULLETS
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("ANOMALY DETECTION SUMMARY")
    print(f"{'='*80}")
    
    if spike_flags:
        print(f"\n🚨 SPIKE ANOMALIES ({len(spike_flags)} flagged days):")
        for flag in spike_flags[:5]:  # Show top 5
            date_str = flag['date'].strftime('%Y-%m-%d')
            print(f"  • {date_str}: Spike Index {flag['spike_index']:.2f} "
                  f"(Requests z={flag['requests_z']:.2f}, Tokens z={flag['tokens_z']:.2f})")
        if len(spike_flags) > 5:
            print(f"  ... and {len(spike_flags) - 5} more")
    else:
        print("\n✓ No spike anomalies detected (all days within threshold)")
    
    if inefficiency_flags:
        print(f"\n⚠️  INEFFICIENCY ANOMALIES ({len(inefficiency_flags)} flagged projects):")
        for flag in inefficiency_flags[:5]:  # Show top 5
            components = (
                f"cost_ratio={flag['cost_ratio']:.2f}, "
                f"premium={flag['premium_share']:.1%}, "
                f"cache={flag['cache_hit_rate']:.1%}"
            )
            print(f"  • {flag['entity']}: Index {flag['inefficiency_index']:.2f} ({components})")
        if len(inefficiency_flags) > 5:
            print(f"  ... and {len(inefficiency_flags) - 5} more")
    else:
        print("\n✓ No inefficiency anomalies detected (all projects within threshold)")
    
    # ========================================================================
    # STEP 6: GENERATE EXECUTIVE SUMMARY
    # ========================================================================
    
    # Build summary
    exec_summary = f"""
[Q4] ANOMALIES & INEFFICIENCIES:
  • Detected {len(spike_flags)} spike event(s) (Spike Index > 3.0)
  • Identified {len(inefficiency_flags)} inefficient project(s) (Inefficiency Index > 1.3)"""
    
    if spike_flags:
        worst_spike = max(spike_flags, key=lambda x: x['spike_index'])
        date_str = worst_spike['date'].strftime('%Y-%m-%d')
        exec_summary += f"\n  • Worst spike: {date_str} (Index: {worst_spike['spike_index']:.2f})"
    
    if inefficiency_flags:
        worst_ineff = max(inefficiency_flags, key=lambda x: x['inefficiency_index'])
        exec_summary += f"\n  • Most inefficient: {worst_ineff['entity']} (Index: {worst_ineff['inefficiency_index']:.2f})"
    
    if not spike_flags and not inefficiency_flags:
        exec_summary += "\n  • All metrics within acceptable thresholds - healthy operation"
    
    print(f"\n{'='*80}")
    print("✓ Q4 Analysis Complete")
    print(f"{'='*80}\n")
    
    return exec_summary.strip()


def analyze_q4_anomalies(df):
    """
    Detect anomalies, spikes, and inefficiencies.
    
    Detects:
    - Request spikes (+30% WoW or z > 3)
    - Tokens/req shifts (±20%)
    - I/O ratio shifts (±20pp)
    - Cache rate changes (±20pp)
    - Model mix swings (±20pp)
    - Inefficiencies (cost ≥1.5x median)
    
    Args:
        df: Input dataframe
    """
    print("\n" + "="*80)
    print("Q4: WHERE ANOMALIES, SPIKES, OR INEFFICIENCIES HAPPEN?")
    print("="*80)
    
    required_cols = ['date']
    missing, available = check_required_columns(df, required_cols)
    
    if missing:
        print(f"❌ Insufficient data for Q4. Missing columns: {missing}")
        return None
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    
    anomalies = []
    
    # ---- 1. REQUEST SPIKES ----
    if 'num_model_requests' in df.columns:
        daily_requests = df.groupby('date')['num_model_requests'].sum().sort_index()
        
        # WoW change
        daily_requests_wow = daily_requests.pct_change(periods=7)
        
        # Z-score (14-day rolling)
        rolling_mean = daily_requests.rolling(window=BASELINE_DAYS, min_periods=1).mean()
        rolling_std = daily_requests.rolling(window=BASELINE_DAYS, min_periods=1).std()
        z_scores = safe_divide(
            daily_requests - rolling_mean,
            rolling_std,
            fill_value=0
        )
        
        # Detect spikes
        spike_dates = daily_requests[
            (daily_requests_wow > SPIKE_THRESHOLD_PCT) | 
            (z_scores > SPIKE_Z_THRESHOLD)
        ].index
        
        for date in spike_dates:
            anomalies.append({
                'date': date,
                'type': 'Request Spike',
                'metric': 'num_model_requests',
                'value': daily_requests.loc[date],
                'z_score': z_scores.loc[date],
                'wow_change': daily_requests_wow.loc[date] if date in daily_requests_wow.index else np.nan
            })
        
        print(f"\n  Found {len(spike_dates)} request spike days")
    
    # ---- 2. INEFFICIENCY BY PROJECT ----
    inefficient_projects = []
    if 'project_id' in df.columns and 'cost_per_1k' in df.columns:
        # Calculate median cost per 1k across all projects
        project_cost_per_1k = df.groupby('project_id')['cost_per_1k'].median()
        org_median = project_cost_per_1k.median()
        
        # Find inefficient projects
        inefficient = project_cost_per_1k[
            project_cost_per_1k >= (INEFFICIENCY_MULTIPLIER * org_median)
        ]
        
        for project, cost in inefficient.items():
            inefficient_projects.append({
                'project_id': project,
                'cost_per_1k': cost,
                'org_median': org_median,
                'inefficiency_ratio': cost / org_median
            })
        
        print(f"  Found {len(inefficient)} inefficient projects (≥{INEFFICIENCY_MULTIPLIER}x median)")
    
    # ---- 3. TOKENS/REQ SHIFTS ----
    if 'tokens_per_req' in df.columns and 'num_model_requests' in df.columns:
        # Weighted daily average
        df['tokens_weighted'] = df['tokens_per_req'] * df['num_model_requests']
        daily_tokens_weighted = df.groupby('date')['tokens_weighted'].sum()
        daily_requests = df.groupby('date')['num_model_requests'].sum()
        daily_avg_tokens = safe_divide(daily_tokens_weighted, daily_requests, 0)
        
        # 3-day MA
        daily_avg_tokens_ma = daily_avg_tokens.rolling(window=MA_WINDOW, min_periods=1).mean()
        
        # Detect shifts (compare to monthly mean)
        monthly_mean = daily_avg_tokens.mean()
        shifts = daily_avg_tokens_ma[
            np.abs(daily_avg_tokens_ma - monthly_mean) / monthly_mean > TOKENS_PER_REQ_SHIFT_PCT
        ]
        
        print(f"  Found {len(shifts)} days with tokens/req shifts")
    
    # ---- 4. CACHE RATE CHANGES ----
    if 'cache_hit_rate' in df.columns and 'cost_usd' in df.columns:
        # Weighted daily cache rate
        df['cache_weighted'] = df['cache_hit_rate'] * df['cost_usd']
        daily_cache_weighted = df.groupby('date')['cache_weighted'].sum()
        daily_cost = df.groupby('date')['cost_usd'].sum()
        daily_cache_rate = safe_divide(daily_cache_weighted, daily_cost, 0)
        
        # Day-over-day change
        cache_dod_change = daily_cache_rate.diff().abs()
        
        cache_changes = cache_dod_change[cache_dod_change > CACHE_CHANGE_PP]
        
        print(f"  Found {len(cache_changes)} days with significant cache rate changes")
    
    # Create visualizations
    fig = plt.figure(figsize=(14, 10))
    
    plot_count = 0
    
    # Plot 1: Daily Spike Index (z-scores)
    if 'num_model_requests' in df.columns:
        plot_count += 1
        ax1 = plt.subplot(3, 1, plot_count)
        
        ax1.bar(z_scores.index, z_scores.values, color='steelblue', alpha=0.7)
        ax1.axhline(y=SPIKE_Z_THRESHOLD, color='red', linestyle='--', 
                   linewidth=1.5, label=f'Threshold (z={SPIKE_Z_THRESHOLD})')
        ax1.axhline(y=-SPIKE_Z_THRESHOLD, color='red', linestyle='--', linewidth=1.5)
        
        ax1.set_ylabel('Z-Score')
        ax1.set_title('Daily Request Spike Index (Z-Score)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
    
    # Plot 2: Inefficiency Index by Project
    if inefficient_projects:
        plot_count += 1
        ax2 = plt.subplot(3, 1, plot_count)
        
        ineff_df = pd.DataFrame(inefficient_projects).sort_values('inefficiency_ratio', ascending=False)
        
        if len(ineff_df) > 10:
            ineff_df = ineff_df.head(10)
        
        bars = ax2.barh(range(len(ineff_df)), ineff_df['inefficiency_ratio'].values, 
                       color='coral', alpha=0.7)
        ax2.set_yticks(range(len(ineff_df)))
        ax2.set_yticklabels(ineff_df['project_id'].values)
        ax2.set_xlabel('Inefficiency Ratio (vs Org Median)')
        ax2.set_title(f'Top Inefficient Projects (Cost/1k ≥ {INEFFICIENCY_MULTIPLIER}x Median)')
        ax2.axvline(x=INEFFICIENCY_MULTIPLIER, color='red', linestyle='--', 
                   linewidth=1.5, label=f'{INEFFICIENCY_MULTIPLIER}x Threshold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.invert_yaxis()
        
        # Add value labels
        for i, ratio in enumerate(ineff_df['inefficiency_ratio'].values):
            ax2.text(ratio, i, f'  {ratio:.2f}x', va='center', fontsize=8)
    
    # Plot 3: Daily metrics overview
    plot_count += 1
    ax3 = plt.subplot(3, 1, plot_count)
    
    if 'tokens_per_req' in df.columns and 'num_model_requests' in df.columns:
        ax3.plot(daily_avg_tokens.index, daily_avg_tokens.values, 
                marker='o', markersize=3, linewidth=1.5, 
                label='Avg Tokens/Req', color='steelblue')
        ax3.set_ylabel('Tokens per Request')
        ax3.set_title('Daily Average Tokens per Request')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    chart_path = f"{ARTIFACTS_DIR}/q4_anomaly_dashboard.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved chart: {chart_path}")
    
    # Save anomaly tables
    if anomalies:
        anomaly_df = pd.DataFrame(anomalies)
        csv_path = f"{ARTIFACTS_DIR}/q4_anomalies.csv"
        anomaly_df.to_csv(csv_path, index=False, float_format='%.4f')
        print(f"✓ Saved anomalies: {csv_path}")
    
    if inefficient_projects:
        ineff_df = pd.DataFrame(inefficient_projects)
        csv_path = f"{ARTIFACTS_DIR}/q4_inefficient_projects.csv"
        ineff_df.to_csv(csv_path, index=False, float_format='%.4f')
        print(f"✓ Saved inefficient projects: {csv_path}")
    
    return {
        'anomalies': anomalies,
        'inefficient_projects': inefficient_projects
    }


# ============================================================================
# EXECUTIVE SUMMARY
# ============================================================================

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
    import os
    if os.path.exists(ARTIFACTS_DIR):
        files = os.listdir(ARTIFACTS_DIR)
        csv_count = len([f for f in files if f.endswith('.csv')])
        png_count = len([f for f in files if f.endswith('.png')])
        print(f"   • Generated {csv_count} CSV tables + {png_count} PNG charts")
    
    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE")
    print("="*80 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("OpenAI COST & USAGE ANALYSIS")
    print("Senior FinOps Data Engineer Tool")
    print("="*80)
    
    # Ask user for monthly budget
    print("\n" + "-"*80)
    print("BUDGET CONFIGURATION")
    print("-"*80)
    budget_input = input(f"Enter your monthly budget in USD (press Enter for default ${MONTHLY_BUDGET:,}): ").strip()
    
    if budget_input:
        try:
            user_budget = float(budget_input.replace(',', ''))
            if user_budget <= 0:
                print(f"⚠️  Invalid budget amount. Using default: ${MONTHLY_BUDGET:,}")
                user_budget = MONTHLY_BUDGET
            else:
                print(f"✓ Budget set to: ${user_budget:,.2f}")
        except ValueError:
            print(f"⚠️  Invalid input. Using default: ${MONTHLY_BUDGET:,}")
            user_budget = MONTHLY_BUDGET
    else:
        user_budget = MONTHLY_BUDGET
        print(f"✓ Using default budget: ${user_budget:,}")
    
    # Ensure artifacts directory exists
    ensure_artifacts_dir()
    
    # Load data with schema report
    df, schema_report = load_data(DATA_FILE)
    
    if df is None:
        print("\n❌ Cannot proceed without data. Please ensure the data file exists.")
        return
    
    # Save schema report to artifacts
    if schema_report:
        import json
        schema_path = f"{ARTIFACTS_DIR}/schema_report.json"
        # Convert datetime objects to strings for JSON serialization
        report_copy = schema_report.copy()
        if report_copy.get('date_range'):
            if report_copy['date_range'].get('min'):
                report_copy['date_range']['min'] = str(report_copy['date_range']['min'])
            if report_copy['date_range'].get('max'):
                report_copy['date_range']['max'] = str(report_copy['date_range']['max'])
        
        with open(schema_path, 'w') as f:
            json.dump(report_copy, f, indent=2)
        print(f"✓ Schema report saved: {schema_path}\n")
    
    # Derive metrics
    df = derive_metrics(df)
    
    # Run analyses
    q1_result = q1_drivers(df)  # Returns string summary
    q2_result = q2_usage_shifts(df)  # Returns string summary
    q3_result = q3_budget(df, user_budget)  # Returns string summary (with user-provided budget)
    q4_result = q4_anomalies(df)  # Returns string summary
    
    # Print executive summary
    print_executive_summary(q1_result, q2_result, q3_result, q4_result)


if __name__ == "__main__":
    main()

