"""
Configuration constants for FinOps OpenAI Analysis.

All thresholds and parameters are defined here for easy tuning.
"""

# ============================================================================
# FILE PATHS
# ============================================================================

DATA_FILE = "OpenAI Cost and Usage Data - June 2025.xlsx"
ARTIFACTS_DIR = "./artifacts"


# ============================================================================
# SPIKE DETECTION THRESHOLDS
# ============================================================================

SPIKE_THRESHOLD_PCT = 0.30      # 30% WoW increase triggers spike flag
SPIKE_Z_THRESHOLD = 3.0         # Z-score threshold (3 = 99.7% confidence)
BASELINE_DAYS = 14              # Rolling baseline window for z-scores


# ============================================================================
# USAGE SHIFT THRESHOLDS
# ============================================================================

TOKENS_PER_REQ_SHIFT_PCT = 0.25   # ±25% sustained shift in tokens/request
IO_RATIO_SHIFT_PP = 0.20          # ±20 percentage points in I/O ratio
CACHE_CHANGE_PP = 0.20            # ±20 pp day-over-day cache hit rate change
MODEL_MIX_SWING_PP = 0.20         # ±20 pp shift in model token share


# ============================================================================
# INEFFICIENCY DETECTION
# ============================================================================

INEFFICIENCY_MULTIPLIER = 1.5     # 1.5x org median = inefficient


# ============================================================================
# BUDGET TRACKING
# ============================================================================

MONTHLY_BUDGET = 50000            # Default monthly budget in USD
DAYS_IN_MONTH = 30                # Assumed days in month


# ============================================================================
# ANALYSIS WINDOWS
# ============================================================================

MA_WINDOW = 3                     # Moving average window (days)


# ============================================================================
# Q1 THRESHOLDS (Cost Driver Classification)
# ============================================================================

VOLUME_THRESHOLD = 1.2            # 1.2x median tokens = volume-driven
MIX_THRESHOLD = 1.3               # 1.3x median premium% = mix-driven
INEFFICIENCY_THRESHOLD = 1.5      # 1.5x median cost/1k = inefficiency-driven


# ============================================================================
# Q2 THRESHOLDS (Usage Pattern Detection)
# ============================================================================

REQUESTS_WOW_THRESHOLD = 0.30     # 30% week-over-week requests change
TOKENS_REQ_THRESHOLD = 0.25       # 25% tokens/req deviation from baseline
CACHE_IO_THRESHOLD = 0.20         # 20pp cache or I/O ratio change
MIX_THRESHOLD = 0.20              # 20pp model mix shift

