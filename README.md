# OpenAI Cost & Usage Analysis Tool

A comprehensive Python analysis tool for OpenAI usage and cost data, designed for FinOps engineers.

## Overview

This tool analyzes OpenAI cost and usage data to answer four critical questions:

1. **Which teams/projects drive spend?**
2. **How do usage patterns change over time?**
3. **Are costs on-budget or at risk?**
4. **Where do anomalies, spikes, or inefficiencies happen?**

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Installation

```bash
pip install -r requirements.txt
```

## Data Requirements

Place your data file in the project root directory:
- **File name**: `OpenAI Cost and Usage Data - June 2025.xlsx`

### Expected Columns

The tool expects the following columns (handles missing columns gracefully):

- `date` - Transaction date
- `project_id` - Project or team identifier
- `user_id` - User identifier
- `api_key_id` - API key used
- `model` - Model name (e.g., gpt-4, gpt-3.5-turbo)
- `service_tier` - Service tier
- `input_tokens` - Number of input tokens
- `output_tokens` - Number of output tokens
- `input_cached_tokens` - Cached input tokens
- `input_uncached_tokens` - Uncached input tokens
- `num_model_requests` - Number of API requests
- `cost_usd` - Cost in USD
- `is_premium_model` - Boolean flag (or inferred from model name)

## Usage

```bash
python main.py
```

## Output

All outputs are saved to the `./artifacts` directory:

### Charts (PNG)

1. **Q1**: Top 5 projects by cost (bar chart)
2. **Q2**: Usage trends over time (small multiples)
   - Daily requests
   - Average tokens per request
   - Cache hit rate
   - Model token share
3. **Q3**: Budget burn-up chart with projections
4. **Q4**: Anomaly dashboard
   - Request spike index
   - Inefficiency index by project

### Data Files (CSV)

1. **Q1**: `q1_top5_projects_summary.csv` - Top projects with metrics
2. **Q2**: `q2_daily_metrics.csv` - Daily time-series metrics
3. **Q3**: `q3_budget_summary.csv` - Budget tracking metrics
4. **Q4**: 
   - `q4_anomalies.csv` - Detected anomaly events
   - `q4_inefficient_projects.csv` - Projects with high unit costs

## Metrics & Thresholds

### Derived Metrics

- `tokens` = input_tokens + output_tokens
- `tokens_per_req` = tokens / num_model_requests
- `io_ratio` = output_tokens / input_tokens
- `cache_hit_rate` = input_cached_tokens / (input_cached_tokens + input_uncached_tokens)
- `cost_per_1k` = 1000 × cost_usd / tokens
- `cost_per_req` = cost_usd / num_model_requests

### Anomaly Detection Thresholds

Configurable in `main.py` constants section:

- **Request Spike**: +30% WoW or z-score > 3 (14-day baseline)
- **Tokens/Req Shift**: ±20-25% sustained (3-day MA)
- **I/O Ratio Shift**: ±20 percentage points vs monthly mean
- **Cache Change**: ±20 percentage points DoD
- **Model Mix Swing**: ±20 percentage points toward premium
- **Inefficiency**: Unit cost ≥ 1.5× org median

### Budget Tracking

- **Monthly Budget**: Default $50,000 (configurable)
- **Pacing Index**: (MTD Cost / Budget) ÷ (Days Elapsed / Days in Month)
- **Projected EOM**: MTD Cost + (7-day avg run rate × days remaining)

## Configuration

Edit constants at the top of `main.py`:

```python
# Data source
DATA_FILE = "OpenAI Cost and Usage Data - June 2025.xlsx"

# Anomaly detection thresholds
SPIKE_THRESHOLD_PCT = 0.30
SPIKE_Z_THRESHOLD = 3.0
TOKENS_PER_REQ_SHIFT_PCT = 0.20
CACHE_CHANGE_PP = 0.20
INEFFICIENCY_MULTIPLIER = 1.5

# Budget parameters
MONTHLY_BUDGET = 50000
DAYS_IN_MONTH = 30
```

## Features

- ✅ Graceful handling of missing columns
- ✅ Division-by-zero protection
- ✅ Modular, documented code
- ✅ Executive summary output
- ✅ Multiple visualization types
- ✅ CSV exports for all analyses
- ✅ Anomaly detection with statistical methods

## Executive Summary

The tool prints a final executive summary with 2-4 key bullets per question, including:

- Top spend drivers with percentages
- Usage trend insights
- Budget status and pacing
- Critical anomalies and inefficiencies

## Error Handling

The tool handles common issues gracefully:

- Missing data file → Error message
- Missing columns → Skip affected analyses, continue with others
- Division by zero → Safe replacement with zero/fill values
- Date parsing errors → Automatic coercion

## License

Internal FinOps tool for cost analysis and optimization.

