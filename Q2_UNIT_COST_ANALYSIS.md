# 💰 Q2 Unit Cost Trend Analysis

## Overview

The Q2 analysis now includes **unit cost trend tracking** that monitors cost per 1K tokens and cost per request over time, automatically identifying pricing and efficiency shifts that are **not driven by volume changes**.

## Why Unit Cost Matters

### The Problem with Volume-Only Analysis

Traditional cost analysis often focuses only on total spend:
```
Total Cost = Volume × Unit Cost
```

If total cost increases, it could be due to:
1. **Volume increase** (more requests) → Expected growth ✅
2. **Unit cost increase** (more expensive per request) → Efficiency problem ❌

**Unit cost analysis isolates efficiency from volume** to identify true optimization opportunities.

---

## What's Tracked

### 🎯 Two Key Metrics

#### 1. Cost per 1K Tokens
```
Cost per 1K = (Total Cost × 1000) ÷ Total Tokens
```
**Measures:** How expensive is each token you process?

**Why it matters:**
- Independent of request volume
- Reflects model choice efficiency
- Shows impact of caching
- Tracks pricing changes

#### 2. Cost per Request
```
Cost per Request = Total Cost ÷ Number of Requests
```
**Measures:** How much does each API call cost?

**Why it matters:**
- Business-level metric (cost per transaction)
- Reflects prompt engineering efficiency
- Shows combined impact of all factors
- Easier for stakeholders to understand

---

## 📊 New Visualizations

### Chart 1: `q2_unit_costs.png` - Time Series

**Single-panel chart with median baseline:**
- **Orange line**: Cost per 1K tokens over time
- **Gray dashed line**: Median cost (baseline reference)
- **X-axis**: Dates clearly displayed
- **Y-axis**: Cost in USD per 1,000 tokens

**What to Look For:**

**Good trends:**
- ↓ Declining unit costs = Efficiency improvements
- → Costs near/below median = Consistent, efficient operations

**Warning signs:**
- ↑ Rising unit costs = Potential efficiency issues
- 📈 Spiky unit costs = Inconsistent model usage
- Costs consistently above median = Room for optimization

---

### Chart 2: `q2_top_unit_cost_changes.png` - Ranked Impact

**Horizontal bar chart showing Top 10 changes:**
- **X-axis**: Percentage change in cost per 1K tokens
- **Y-axis**: Date of change
- **Bar colors**: Root cause type
  - 🟠 Orange: Model Mix Impact
  - 💗 Pink: Prompt Length Impact
  - 💜 Purple: Caching Impact
  - 💛 Yellow: Pricing/Efficiency Shift
- **Labels on bars**: Change %, trigger type, cost value

**Purpose:**
- Instantly see the biggest cost shifts
- Identify which dates had the most impact
- Compare magnitude across different root causes

**Example:**
```
2025-06-15  |████████████████████████| +35.2% | Model Mix Impact
                                        $0.0245/1K

2025-06-25  |██████████████| -28.3% | Caching Impact
                              $0.0125/1K
```

---

## 🔍 Automated Root Cause Detection

When unit costs shift by ±20% week-over-week, the system automatically diagnoses the cause:

### 1. 🔄 Model Mix Impact
**Detected when:**
- Unit cost changes ±20%
- Model token share changed >10%

**What it means:**
- Shifted from cheaper to more expensive models (or vice versa)
- Example: Moving from GPT-3.5 to GPT-4

**Console output:**
```
UNIT COST Changes (Top 5 of 12 - ranked by impact):
  • 2025-06-15: Model Mix Impact (+35.2%)
    → Cost/1K increased 35.2%, likely due to model mix shift
    → Cost/1K: $0.0245, Cost/Req: $0.1523
    🔄 Model choice changed - affecting unit economics
```

---

### 2. 📝 Prompt Length Impact
**Detected when:**
- Unit cost changes ±20%
- Tokens per request changed >15%
- Model mix stayed stable

**What it means:**
- Prompts got longer (or shorter)
- More (or fewer) examples in prompts
- Output length changed

**Console output:**
```
UNIT COST Changes (1 event(s)):
  • 2025-06-20: Prompt Length Impact
    → Cost/1K increased 22.5%, correlated with token/req change
    → Cost/1K: $0.0189, Cost/Req: $0.2145
    📝 Prompt/output length changed - affecting efficiency
```

---

### 3. 💾 Caching Impact
**Detected when:**
- Unit cost changes ±20%
- Cache hit rate changed >15%
- Model mix and tokens/req stayed stable

**What it means:**
- Cache utilization improved (or degraded)
- Cached tokens are cheaper or free

**Console output:**
```
UNIT COST Changes (1 event(s)):
  • 2025-06-25: Caching Impact
    → Cost/1K decreased 28.3%, correlated with cache rate change
    → Cost/1K: $0.0125, Cost/Req: $0.0875
    💾 Cache utilization changed - affecting costs
```

---

### 4. 💰 Pricing/Efficiency Shift
**Detected when:**
- Unit cost changes ±20%
- No correlating changes in model mix, tokens/req, or cache

**What it means:**
- Provider pricing change
- API tier change
- Complex multi-factor change
- Requires manual investigation

**Console output:**
```
UNIT COST Changes (1 event(s)):
  • 2025-06-28: Pricing/Efficiency Shift
    → Cost/1K increased 24.1%, not volume-driven
    → Cost/1K: $0.0235, Cost/Req: $0.1890
    💰 Pricing or efficiency shift - not driven by volume alone
```

---

## 📈 Executive Summary Integration

The executive summary now includes unit cost metrics:

```
[Q2] USAGE TRENDS:
  • Analyzed 30 days: 2025-06-01 to 2025-06-30
  • Total requests: 1,250,000 (avg 41,667/day)
  • Detected shifts: 2 I/O ratio shift(s), 3 unit cost shift(s), 1 cache change(s)
  • Average tokens/req: 4,523, output/input ratio: 0.42, cache hit rate: 68.5%
  • Unit costs: $0.0185/1K tokens, $0.1245/request
  • Largest unit cost shift: 35.2% increase (Model Mix Impact)
```

---

## 🎯 Use Cases

### 1. Distinguish Volume Growth from Inefficiency

**Scenario:** Total cost increased 50% month-over-month

**Analysis:**
- Check requests: +45% ✅ (business growth)
- Check unit costs: +3% ⚠️ (minor efficiency issue)

**Conclusion:** Growth-driven, not a cost problem. Minor optimization opportunity.

---

### 2. Track Optimization Initiatives

**Scenario:** Engineering team optimized prompts

**Before:**
- Cost per 1K: $0.0250
- Cost per req: $0.2000

**After:**
- Cost per 1K: $0.0180 (↓28%)
- Cost per req: $0.1440 (↓28%)

**Result:** Clear evidence of successful optimization!

---

### 3. Detect Unintended Model Changes

**Scenario:** Cost per 1K suddenly jumped 80%

**Root cause detection:**
- Diagnosis: "Model Mix Impact"
- GPT-3.5 share: 80% → 20%
- GPT-4 share: 20% → 80%

**Action:** Investigate deployment - was this intentional?

---

### 4. Validate Caching Impact

**Scenario:** Enabled prompt caching

**Before:**
- Cost per 1K: $0.0200
- Cache hit rate: 0%

**After:**
- Cost per 1K: $0.0140 (↓30%)
- Cache hit rate: 65%

**Diagnosis:** "Caching Impact"
**Result:** Quantified cache savings!

---

## ⚙️ Configuration

### Detection Threshold

```python
# main.py, line ~1591
if abs(cost_1k_change) > 0.20:  # 20% change triggers analysis
```

**Adjust based on:**
- Data volatility
- Business requirements
- Noise levels in your environment

### Root Cause Thresholds

```python
# Model mix sensitivity
if abs(curr_share - prev_share) > 0.10:  # 10% model share change

# Tokens per request sensitivity
if abs(tpr_change) > 0.15:  # 15% tokens/req change

# Cache rate sensitivity
if abs(curr_cache - prev_cache) > 0.15:  # 15% cache rate change
```

---

## 📊 Files Generated

### Charts
- **`artifacts/q2_unit_costs.png`** - Time series dual-line chart with dates on X-axis
- **`artifacts/q2_top_unit_cost_changes.png`** - Horizontal bar chart of Top 10 changes, ranked by impact

### CSV
- **`artifacts/q2_daily_metrics.csv`** - Now includes:
  - `cost_per_1k` - Cost per 1,000 tokens
  - `cost_per_req` - Cost per request

---

## 💡 Interpretation Guide

### Scenario 1: Both Metrics Rise Together
```
Cost per 1K: ↑
Cost per Req: ↑
```
**Likely cause:** Model mix shifted to premium models
**Action:** Review model routing logic

---

### Scenario 2: Cost per Req Rises, Cost per 1K Stable
```
Cost per 1K: →
Cost per Req: ↑
```
**Likely cause:** Prompts got longer (more tokens per request)
**Action:** Review prompt engineering - is the extra context necessary?

---

### Scenario 3: Both Metrics Decline
```
Cost per 1K: ↓
Cost per Req: ↓
```
**Likely cause:** Efficiency improvement (caching, cheaper models, shorter prompts)
**Action:** Document and celebrate! 🎉

---

### Scenario 4: Cost per Req Declines, Cost per 1K Stable
```
Cost per 1K: →
Cost per Req: ↓
```
**Likely cause:** Prompts got shorter (fewer tokens per request)
**Action:** Verify output quality maintained

---

## 🚀 Running the Analysis

No configuration needed! Just run:

```bash
python main.py
```

The enhanced Q2 analysis automatically:
- Calculates unit costs daily
- Detects significant changes (±20% WoW)
- Diagnoses root causes
- Generates visualization
- Includes in executive summary

---

## 📝 Example Output

```
================================================================================
Q2: HOW USAGE PATTERNS CHANGE OVER TIME?
================================================================================

  Aggregating daily metrics...
  ✓ Aggregated 30 days of data
  ✓ Date range: 2025-06-01 to 2025-06-30

  Detecting usage shifts and triggers...
  ✓ Detected 12 trigger events

  Creating visualizations...
  ✓ Saved: artifacts/q2_requests.png
  ✓ Saved: artifacts/q2_tokens_per_req.png
  ✓ Saved: artifacts/q2_io_ratio.png
  ✓ Saved: artifacts/q2_unit_costs.png              ← NEW!
  ✓ Saved: artifacts/q2_top_unit_cost_changes.png   ← NEW!
  ✓ Saved: artifacts/q2_cache_rate.png
  ✓ Saved: artifacts/q2_model_mix.png

================================================================================
USAGE SHIFT DIAGNOSIS
================================================================================

UNIT COST Changes (3 event(s)):              ← NEW!
  • 2025-06-15: Model Mix Impact
    → Cost/1K increased 35.2%, likely due to model mix shift
    → Cost/1K: $0.0245, Cost/Req: $0.1523
    🔄 Model choice changed - affecting unit economics

  • 2025-06-20: Prompt Length Impact
    → Cost/1K increased 22.5%, correlated with token/req change
    → Cost/1K: $0.0189, Cost/Req: $0.2145
    📝 Prompt/output length changed - affecting efficiency

  • 2025-06-25: Caching Impact
    → Cost/1K decreased 28.3%, correlated with cache rate change
    → Cost/1K: $0.0125, Cost/Req: $0.0875
    💾 Cache utilization changed - affecting costs
```

---

## ✅ Benefits

### Immediate
1. **Separate volume from efficiency** - Know if costs rose due to growth or waste
2. **Automatic root cause** - No manual detective work
3. **Quantify optimizations** - Prove ROI of engineering work
4. **Catch regressions** - Alert when efficiency degrades

### Strategic
1. **Inform pricing** - Understand unit economics for customer pricing
2. **Track SLAs** - Monitor cost per transaction commitments
3. **Guide investments** - Prioritize optimization by impact
4. **Enable forecasting** - Model costs based on volume + unit cost trends

---

## 🎓 Key Concepts

### Unit Cost vs Total Cost

| Metric | Formula | What it Shows | Use For |
|--------|---------|---------------|---------|
| **Total Cost** | Sum of all charges | Overall spend | Budget tracking |
| **Unit Cost** | Cost ÷ Volume | Efficiency per unit | Optimization |

**Example:**
- Week 1: 1M requests × $0.10/req = $100K
- Week 2: 2M requests × $0.10/req = $200K
- **Analysis:** Unit cost stable ($0.10), cost doubled due to volume (good!)

vs.

- Week 1: 1M requests × $0.10/req = $100K
- Week 2: 1M requests × $0.15/req = $150K
- **Analysis:** Same volume, unit cost rose 50% (investigate!)

---

## 🎊 Summary

**What you get:**
✅ Cost per 1K tokens tracking
✅ Cost per request tracking
✅ Automatic root cause detection (4 scenarios)
✅ Visual timeline with flags
✅ Executive summary integration
✅ Clear distinction between volume and efficiency

**Lines of code:** ~75 (excluding detection logic)
**Charts added:** 1 (`q2_unit_costs.png`)
**Root causes detected:** 4 types

---

**Status:** ✅ Complete and ready to use!

