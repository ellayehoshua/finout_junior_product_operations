"""Simple test for utility functions"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("Starting test...")

from main import save_fig, export_table, format_axis_labels, ensure_artifacts_dir

print("Imports successful!")

# Ensure artifacts dir
ensure_artifacts_dir()
print("Artifacts directory ready")

# Test 1: export_table
print("\nTest 1: export_table")
df = pd.DataFrame({'a': [1.234, 2.345], 'b': [3.456, 4.567]})
export_table(df, 'simple_test.csv')

# Test 2: save_fig
print("\nTest 2: save_fig")
plt.figure()
plt.plot([1, 2, 3], [4, 5, 6])
plt.title('Test')
save_fig('simple_test.png')

# Test 3: format_axis_labels
print("\nTest 3: format_axis_labels")
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [4, 5, 6])
format_axis_labels(ax, 'X', 'Y', 'Test Chart')
save_fig('simple_test_formatted.png')

print("\nAll tests complete!")

