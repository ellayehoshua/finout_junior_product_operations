"""
Utility functions for FinOps OpenAI Analysis.

Provides reusable helper functions for:
- Directory management
- Safe mathematical operations
- String formatting
- Chart/table export standardization
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from config import ARTIFACTS_DIR


def ensure_artifacts_dir():
    """Create artifacts directory if it doesn't exist."""
    if not os.path.exists(ARTIFACTS_DIR):
        os.makedirs(ARTIFACTS_DIR)
        print(f"✓ Created directory: {ARTIFACTS_DIR}")


def safe_divide(numerator, denominator, fill_value=0):
    """
    Safely divide two arrays/series, handling division by zero.
    
    Args:
        numerator: Numerator values
        denominator: Denominator values
        fill_value: Value to use when denominator is 0 (default: 0)
        
    Returns:
        Division result with fill_value where denominator is 0
    """
    return np.where(denominator != 0, numerator / denominator, fill_value)


def to_snake_case(name):
    """
    Convert column name to snake_case.
    
    Args:
        name: Original column name
        
    Returns:
        snake_case version of the name
    """
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

