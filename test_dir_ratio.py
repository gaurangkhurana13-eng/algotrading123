import numpy as np
import pandas as pd

# Let's test the directional ratio on probability matrix
np.random.seed(42)
# Suppose probs has shape (717, 3) where columns are [P(0), P(1), P(2)]
p0 = np.random.uniform(0.25, 0.40, size=717)
p1 = np.random.uniform(0.30, 0.42, size=717)
p2 = np.random.uniform(0.25, 0.40, size=717)
totals = p0 + p1 + p2
probs = np.column_stack([p0/totals, p1/totals, p2/totals])

# Method A: Raw probability >= 0.50
raw_long = probs[:, 2] >= 0.50
print("Method A (Raw >= 0.50):", np.sum(raw_long))

# Method B: Directional Ratio P(2) / (P(2) + P(0)) >= 0.52 and P(2) > P(1)
dir_ratio = probs[:, 2] / (probs[:, 2] + probs[:, 0] + 1e-12)
dir_long = (dir_ratio >= 0.52) & (probs[:, 2] > probs[:, 1])
print("Method B (Directional Ratio >= 0.52 & > Chop):", np.sum(dir_long))

# Method C: Relative Edge over baseline (P(2) >= 0.36)
edge_long = probs[:, 2] >= 0.36
print("Method C (Raw P(2) >= 0.36):", np.sum(edge_long))
