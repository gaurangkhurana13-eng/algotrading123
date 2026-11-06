import pickle

with open('data/universe_78_15m_1y.pkl', 'rb') as f:
    d = pickle.load(f)

sym = list(d.keys())[0]
print("Sample symbol:", sym)
print("Type of dict value:", type(d[sym]))
print("Columns:", d[sym].columns.tolist())
print("Index:", type(d[sym].index), d[sym].index.name)
print("\nSample Head:")
print(d[sym].head(2))
