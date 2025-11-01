"""
Data loading and metric derivation for FinOps OpenAI Analysis.

Handles:
- Excel file loading with auto-sheet detection
- Column name normalization (snake_case)
- Date parsing and validation
- Schema reporting
- Derived metric calculations
"""

import pandas as pd
import numpy as np
from utils import safe_divide, to_snake_case


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


def load_data(file_path):
    """
    Load OpenAI cost and usage data from Excel file with enhanced features.
    
    Features:
    - Auto-detects sheet (uses first non-empty sheet)
    - Converts column names to snake_case
    - Parses date column to datetime and creates 'day' column (date only)
    - Reports found vs missing expected columns
    - Returns DataFrame and schema_report dict
    
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
        print(f"LOADING DATA")
        print(f"{'='*80}")
        print(f"File: {file_path}")
        
        # Read Excel file and detect sheets
        xl_file = pd.ExcelFile(file_path)
        print(f"  Found {len(xl_file.sheet_names)} sheet(s): {', '.join(xl_file.sheet_names)}")
        
        # Auto-detect: use first sheet with data
        sheet_to_use = None
        for sheet_name in xl_file.sheet_names:
            df_test = pd.read_excel(file_path, sheet_name=sheet_name, nrows=5)
            if len(df_test) > 0 and len(df_test.columns) > 0:
                sheet_to_use = sheet_name
                print(f"  ✓ Auto-selected sheet: '{sheet_name}'")
                break
        
        if sheet_to_use is None:
            print(f"❌ ERROR: No valid sheet found with data")
            return None, None
        
        # Read the selected sheet
        df = pd.read_excel(file_path, sheet_name=sheet_to_use)
        
        print(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")
        
        # Store original column names for reference
        original_columns = df.columns.tolist()
        
        # Convert column names to snake_case
        df.columns = [to_snake_case(col) for col in df.columns]
        
        print(f"\n  Column name normalization:")
        for orig, snake in zip(original_columns, df.columns):
            if orig != snake:
                print(f"    '{orig}' → '{snake}'")
        
        # Parse date column to datetime
        date_candidates = ['date', 'timestamp', 'datetime', 'created_at', 'created']
        date_column = None
        
        for candidate in date_candidates:
            if candidate in df.columns:
                date_column = candidate
                break
        
        if date_column:
            print(f"\n  Date handling:")
            print(f"    Found date column: '{date_column}'")
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            
            # Create 'day' column (date only, no time)
            df['day'] = df[date_column].dt.date
            
            # Count invalid dates
            invalid_dates = df[date_column].isna().sum()
            if invalid_dates > 0:
                print(f"    ⚠️  Warning: {invalid_dates} rows with invalid dates")
            
            print(f"    ✓ Parsed '{date_column}' to datetime")
            print(f"    ✓ Created 'day' column (date only)")
            
            # Print date range
            if df[date_column].notna().any():
                min_date = df[date_column].min()
                max_date = df[date_column].max()
                print(f"    Date range: {min_date.date()} to {max_date.date()}")
        else:
            print(f"\n  ⚠️  Warning: No date column found (checked: {', '.join(date_candidates)})")
        
        # Schema report: found vs missing columns
        print(f"\n  Schema Report:")
        print(f"  {'-'*76}")
        
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
            'sheet_name': sheet_to_use,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'found_columns': found_columns,
            'missing_columns': missing_columns,
            'extra_columns': extra_columns,
            'has_date_column': date_column is not None,
            'date_column': date_column,
            'date_range': {
                'min': df[date_column].min() if date_column and df[date_column].notna().any() else None,
                'max': df[date_column].max() if date_column and df[date_column].notna().any() else None
            } if date_column else None,
            'column_mapping': dict(zip(original_columns, df.columns))
        }
        
        print(f"\n  {'='*76}")
        print(f"  ✓ Data loaded successfully")
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

