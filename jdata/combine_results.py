import pandas as pd

path = "raw_data/results/try_1/Result_%s.csv"
resultNum = 3
dfs = []
for i in range(0, resultNum, 1):
    df = pd.read_csv(path % i)
    dfs.append(df)

concat_df = pd.concat(dfs,ignore_index=True)
concat_df['new_code'] = concat_df.index + 1
concat_df['trans_code'] = concat_df['new_code'].apply(lambda x: "DP%04d" % x)
concat_df.drop(['new_code'], axis=1, inplace=True)
print("total costs: %.2f" % concat_df['total_cost'].sum())
concat_df.to_csv("raw_data/results/try_1/Result.csv", index=False)

