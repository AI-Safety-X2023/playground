from __future__ import annotations

import pandas as pd

# https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#evaluation-order-matters
pd.options.mode.copy_on_write = True