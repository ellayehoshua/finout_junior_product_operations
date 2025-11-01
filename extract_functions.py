"""Helper script to extract function line ranges from main.py"""

with open('main.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find function start lines
functions = {
    'q1_drivers': None,
    'analyze_q1_project_spend': None,
    'q2_usage_shifts': None,
    'q3_budget': None,
    'q4_anomalies': None,
    'analyze_q4_anomalies': None,
    'print_executive_summary': None,
    'main': None
}

for i, line in enumerate(lines, 1):
    for func in functions:
        if line.strip().startswith(f'def {func}'):
            functions[func] = i

print("Function line numbers:")
for func, line_num in functions.items():
    if line_num:
        print(f"  {func}: line {line_num}")

