import test_exact
import json

res = test_exact.query_brokerage_exact(37.0, 37.5, 35, product="I", sec_id=12018)
print(json.dumps(res, indent=2))
