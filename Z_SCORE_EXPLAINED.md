# 📊 Z-Score Explained (In Simple Terms)

## 🤔 What is a Z-Score?

Think of a z-score as a **"weirdness detector"**. It tells you how unusual a day is compared to your normal behavior.

### 🎯 **The Simplest Explanation:**

**Z-score = How many "steps" away from normal is this day?**

---

## 📏 Real-World Analogy

Imagine you normally spend **$1,000/day** on OpenAI:

| Day | Spending | What's Happening? | Z-Score |
|-----|----------|-------------------|---------|
| Monday | $1,000 | Normal day | **0** (perfectly normal) |
| Tuesday | $1,500 | A bit high | **1** (1 step above normal) |
| Wednesday | $2,000 | Getting unusual | **2** (2 steps above normal) |
| Thursday | $5,000 | **ALERT!** Very unusual | **3** (3 steps above normal) 🚨 |
| Friday | $10,000 | **CRISIS!** Something broke | **6** (6 steps above normal) 🔥 |

---

## 🔢 The Math (But Simpler)

```
Z-Score = (Today's Value - Average) ÷ Normal Variation

Example:
- Your average daily spending: $1,000
- Normal variation: $500
- Today you spent: $2,500

Z-Score = ($2,500 - $1,000) ÷ $500 = 3.0
```

**Translation:** Today you spent **3x more** than your typical variation!

---

## 🚦 What Do Z-Scores Mean?

| Z-Score | Meaning | Should You Worry? |
|---------|---------|-------------------|
| **0** | Perfectly normal | ✅ No |
| **+1** | A bit higher than usual | ✅ No, this is normal variation |
| **+2** | Noticeably higher | 🟡 Maybe check it out |
| **+3** | Very unusual (happens ~0.3% of the time) | 🔴 **YES! Investigate!** |
| **+4 or more** | Extremely rare! | 🔥 **ALERT! Something is wrong!** |

---

## 🎯 Why is the Threshold 3.0?

### **Statistical Reason:**
In a normal system, values beyond z-score = 3 happen **only 0.3% of the time** (3 out of 1,000 days).

**Translation:** If you see a z-score of 3+, there's a **99.7% chance** something unusual is happening!

### **Practical Reason:**
- **Z-score < 3:** Normal business fluctuations
- **Z-score ≥ 3:** Real problems that need attention

**It's like a smoke detector:** Set too sensitive (z=1) and it goes off when you cook. Set at z=3, it only goes off for real fires! 🔥

---

## 📊 Your Token Spike Index

### **What We're Measuring:**
The Token Spike Index combines **two z-scores**:
1. **Requests z-score:** How unusual is today's number of requests?
2. **Tokens z-score:** How unusual is today's token usage?

**Formula:**
```
Token Spike Index = (Requests Z-Score + Tokens Z-Score) ÷ 2
```

### **Why Both?**
Sometimes you have:
- **More requests** but same tokens/request → Volume spike
- **Same requests** but way more tokens/request → Prompt spike

We average both to catch either type of anomaly!

---

## 🔍 Why Are Specific Days Flagged?

Looking at your chart, here's what happened:

### **June 12-13: MAJOR SPIKE (z ≈ 3.0+)**

**What the numbers show:**
- June 12: ~18.5M requests (**+79.5% WoW**)
- June 13: ~76M requests (**+582% WoW!**)

**Why it's flagged:**
```
Normal day: 10M requests
June 13: 76M requests
Typical variation: 5M requests

Z-Score = (76M - 10M) ÷ 5M = 13.2 🔥🔥🔥

This is MASSIVELY above the threshold of 3!
```

**Translation:** June 13 had **13x more variation** than normal. This is like your smoke detector screaming!

---

## 🧮 How We Calculate It

### **Step 1: Calculate the 14-day rolling average**
Look at the past 14 days to understand "normal"

Example:
- Days 1-14 average: 10M requests/day
- Typical variation: 2M requests

### **Step 2: Compare today to that average**
- Day 15: 16M requests
- Difference: 16M - 10M = 6M above normal
- Z-score: 6M ÷ 2M = **3.0** ⚠️

**Flag it!** This is 3 standard deviations above normal.

---

## 🎯 Why This Matters for Your Organization

### **Without Z-Score Flagging:**
You'd need to manually check every day and guess what's "too high"
- Is 15M requests high? 
- What about 20M?
- How do I know if it's just Monday being busy or a real problem?

### **With Z-Score Flagging:**
The system **automatically knows your normal patterns** and alerts you only when something is truly unusual:
- ✅ Automatically adjusts to your baseline
- ✅ Accounts for natural day-to-day variation
- ✅ Only alerts on REAL anomalies (not just "high" days)

---

## 📈 Real Example from Your Data

### **June 13 Breakdown:**

**Requests:**
- Your normal: 11.5M requests/day
- June 13: 76M requests
- Z-score: ~**13** (WAY above threshold!)

**Tokens:**
- Your normal: 15B tokens/day
- June 13: Also spiked dramatically
- Z-score: ~**12**

**Token Spike Index:**
```
(13 + 12) ÷ 2 = 12.5 🔥

This is 4x higher than our threshold of 3!
```

**What this means:** 
June 13 was an **extreme outlier**. This wasn't just a busy day—something fundamentally different happened (maybe a new product launch, bot traffic, or a system error).

---

## 🎓 Key Takeaways

1. **Z-Score = How unusual is today?**
   - 0 = Normal
   - 3 = Very unusual (flag it!)

2. **Threshold of 3 = Industry standard**
   - Statistically sound (99.7% confidence)
   - Practically useful (reduces false alarms)

3. **Flagged days = Real problems**
   - Not just "high" days
   - Days that are **statistically abnormal** for YOUR organization

4. **Automatic adjustment**
   - Learns YOUR patterns
   - No manual threshold setting needed
   - Works even as your usage grows

---

## 💡 Think of It This Way:

**Without z-scores:**
"Today we had 20M requests. Is that bad?"
→ You don't know without context!

**With z-scores:**
"Today we had a z-score of 5.0"
→ **YES, that's bad!** This is far beyond normal, investigate immediately!

---

## 🚀 Bottom Line

**Z-Score = Your AI usage's "check engine light"**
- Threshold of 3 = Serious warning (not just a yellow light)
- Flagged spikes = Something needs your attention NOW

Simple! 🎯





