# 📊 Q2 Output-to-Input Ratio Analysis

## Overview

The Q2 analysis now includes **intelligent output-to-input ratio tracking** that automatically detects and interprets changes in how your AI systems use tokens.

## What's New

### 🎯 New Visualization: `q2_io_ratio.png`

A single-panel chart showing:

**Input vs Output Tokens per Request**
- Green line (left Y-axis): Input tokens per request
- Yellow line (right Y-axis): Output tokens per request
- Dual Y-axes to accommodate different scales
- Dates clearly displayed on X-axis
- Markers on each data point for easy tracking

## Scenario Detection Logic

The analysis automatically identifies **4 key scenarios** based on week-over-week changes:

### 1. 🚀 Prompt Trimming
**Conditions:**
- Output/Input ratio increases (>15%)
- Input tokens decline (>10%)
- Output tokens remain stable (±10%)

**What it means:**
- Prompts were streamlined through trimming
- Fewer examples or shorter system prompts
- **Efficiency gain** ⚡

---

### 2. 📝 Verbose Outputs
**Conditions:**
- Output/Input ratio increases (>15%)
- Output tokens rise (>10%)
- Input tokens remain stable (±10%)

**What it means:**
- Answers became more detailed
- Likely due to instruction change
- May indicate reporting/explanation mode enabled

---

### 3. 📚 Context Expansion
**Conditions:**
- Output/Input ratio decreases (>15%)
- Input tokens rise (>10%)
- Output tokens remain stable (±10%)

**What it means:**
- Inputs expanded with additional context
- Larger retrieval window
- More examples in few-shot prompts
- Enhanced RAG context

---

### 4. ✂️ Stricter Outputs
**Conditions:**
- Output/Input ratio decreases (>15%)
- Output tokens fall (>10%)

**What it means:**
- Responses constrained to be more concise
- JSON format enforced
- Explicit length limits added
- Token budgets implemented

---

### 5. ⚙️ I/O Balance Shift
**Conditions:**
- Ratio changes >15% but doesn't fit above patterns

**What it means:**
- Both input and output changed
- Complex scenario requiring manual review

## Output Format

### Console Output

When I/O ratio shifts are detected, you'll see:

```
I/O RATIO Changes (3 event(s)):
  • 2025-06-15: Prompt Trimming
    → Input streamlined (↓22.5%), output stable, ratio ↑28.3%
    ⚡ Efficiency gain: Prompts streamlined through trimming/fewer examples

  • 2025-06-20: Context Expansion
    → Input expanded (↑35.2%), output stable, ratio ↓26.1%
    📈 Context growth: Additional context/retrieval added to inputs

  • 2025-06-25: Stricter Outputs
    → Output constrained (↓18.7%), ratio ↓22.4%
    ✅ Output constraint: Responses limited (e.g., JSON format, length limits)
```

### Executive Summary

The executive summary now includes:
```
[Q2] USAGE TRENDS:
  • Analyzed 30 days: 2025-06-01 to 2025-06-30
  • Total requests: 1,250,000 (avg 41,667/day)
  • Detected shifts: 2 prompt shift(s), 3 I/O ratio shift(s), 1 cache change(s)
  • Average tokens/req: 4,523, output/input ratio: 0.42, cache hit rate: 68.5%
```

## Technical Implementation

### New Metrics Calculated

```python
# Separate tracking of input/output per request
daily_metrics['input_tokens_per_req'] = total_input / total_requests
daily_metrics['output_tokens_per_req'] = total_output / total_requests
daily_metrics['io_ratio'] = output / input
```

### Detection Algorithm

1. **Week-over-Week Comparison**: Compares each day to 7 days prior
2. **Threshold**: 15% change in ratio triggers analysis
3. **Component Analysis**: Examines input and output changes
4. **Scenario Matching**: Pattern matches to one of 4 scenarios
5. **Flag Creation**: Stores scenario with metadata

### Colors Used

- `COLOR_PRIMARY` (Green): Input tokens line (left Y-axis)
- `COLOR_WARNING` (Yellow): Output tokens line (right Y-axis)

## Use Cases

### 1. Tracking Prompt Engineering Efforts
Monitor the impact of prompt optimization initiatives:
- See immediate effect of trimming prompts
- Validate that outputs remain stable while inputs shrink

### 2. Detecting Unintended Changes
Catch configuration drift:
- Alert when outputs become unexpectedly verbose
- Identify when retrieval window grew without authorization

### 3. Cost Optimization
Quantify efficiency improvements:
- Prompt trimming directly reduces input costs
- Stricter outputs reduce output costs

### 4. Quality Assurance
Ensure changes don't degrade responses:
- Verify output quality maintained during trimming
- Confirm expansion provided intended context

## Configuration

Thresholds are configurable in the detection logic:

```python
# main.py, line ~1510
if abs(io_change_pct) > 0.15:  # 15% change threshold
    
    if io_change_pct > 0 and input_change_pct < -0.10:  # 10% input decline
        scenario = 'Prompt Trimming'
```

Adjust these values based on your data characteristics and noise levels.

## Files Generated

- **Chart**: `artifacts/q2_io_ratio.png` - Visual analysis
- **Data**: `artifacts/q2_daily_metrics.csv` - Includes `io_ratio`, `input_tokens_per_req`, `output_tokens_per_req`

## Example Interpretation

**Scenario**: You see a "Prompt Trimming" flag on June 15

**Investigation Steps**:
1. Check code deployments on/around June 15
2. Review prompt engineering changes
3. Validate output quality metrics remained stable
4. Calculate cost savings: (old_input - new_input) × requests × cost_per_token
5. Document as efficiency win if quality maintained

**Expected Business Impact**:
- Immediate cost reduction on input tokens
- Improved latency (fewer tokens to process)
- Potential template for other prompts

---

## Summary

This enhancement provides **automatic storytelling** about how your AI token usage evolves. Instead of just showing numbers, it explains *why* ratios changed and *what* that means for your operations.

**Key Benefits:**
- ✅ Automatic scenario detection
- ✅ Visual flags on charts
- ✅ Actionable interpretations
- ✅ Cost optimization insights
- ✅ Quality monitoring

