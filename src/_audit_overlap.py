import pandas as pd

train = pd.read_csv(r'd:\WYNTRIX - INTERNSHIP\new\outputs\processed\train_clean.csv')
val   = pd.read_csv(r'd:\WYNTRIX - INTERNSHIP\new\outputs\processed\val_clean.csv')
test  = pd.read_csv(r'd:\WYNTRIX - INTERNSHIP\new\outputs\processed\test_clean.csv')

train_s = set(train['sentence'].astype(str))
val_s   = set(val['sentence'].astype(str))
test_s  = set(test['sentence'].astype(str))

tv_overlap = train_s.intersection(val_s)
tt_overlap = train_s.intersection(test_s)

print(f'Train rows    : {len(train):,}')
print(f'Val rows      : {len(val):,}')
print(f'Test rows     : {len(test):,}')
print()
print(f'Train vs Val  overlap: {len(tv_overlap)} sentence(s)')
for s in sorted(tv_overlap):
    tl = train[train["sentence"]==s]["label"].values
    vl = val[val["sentence"]==s]["label"].values
    print(f'  "{s}"  train_label={tl}  val_label={vl}')
print()
print(f'Train vs Test overlap: {len(tt_overlap)} sentence(s)')
for s in sorted(tt_overlap):
    print(f'  "{s}"')
