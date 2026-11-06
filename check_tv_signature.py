import inspect
from tvDatafeed import TvDatafeed

sig = inspect.signature(TvDatafeed.get_hist)
print("TvDatafeed.get_hist signature:", sig)
doc = inspect.getdoc(TvDatafeed.get_hist)
print("Docstring:", doc)
