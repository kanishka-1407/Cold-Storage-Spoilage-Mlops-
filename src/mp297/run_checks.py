import pprint
import numpy as np
import pandas as pd
from mp297.data_checks import *

pp = pprint.PrettyPrinter(width=110)
train = pd.read_csv("data/raw/train.csv")
hold = pd.read_csv("data/raw/holdout_features.csv")
later = pd.read_csv("data/raw/later.csv")

print("Rows:", len(train))
print("\n== 1 LEAK =="); pp.pprint(check_leak(train, hold))
print("\n== 7 SPELLING train =="); pp.pprint(check_spelling(train))
print("\n== 7 SPELLING later (clean reference) =="); pp.pprint(check_spelling(later))
print("\n== 3 SENTINELS =="); pp.pprint(check_sentinels(train))
print("\n== 4 DUPLICATES =="); pp.pprint(check_duplicates(train))
print("\n== 5 STUCK (top 3) =="); pp.pprint(check_stuck(train)[:3])
print("\n== 6 TARGET TYPOS =="); pp.pprint(check_target_typos(train))

# Re-check 1 and 2 after setting the problems found above aside
no_typo = train[train[TARGET] < 30]
print("\n== 1b LEAK corr without typo rows ==",
      round(no_typo["spoilage_found_at_unloading"].corr(no_typo[TARGET]), 4))
tmp = train.replace({"power_cuts_hours": {9999: np.nan}})
print("\n== 2 UNIT MIXUP (sentinels removed) ==")
r = check_unit_mixup(tmp); print(r["group_means"]); pp.pprint(r["max_over_min"])