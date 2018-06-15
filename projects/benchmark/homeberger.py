import pandas as pd

for size in range(400, 1001, 200):
    base_path = "data/homberger/%s_customer_instances/" % size
    # print(base_path)
    bks_path = base_path + "homberger_%s_bks.txt" % size
    # print(bks_path)
    bks = pd.read_table(bks_path, sep="\t")
    # print(bks)
    jsprit_path = base_path + "jsprit_homberger_%s_res.txt" % size
    jsprit = pd.read_table(jsprit_path, sep="\t")
    # print(jsprit)
    jsprit['Example'] = jsprit['Example'].str.lower()
    merge = pd.merge(bks, jsprit, on='Example', how='left')
    merge['Costs'] = merge['Costs'].map(lambda x: round(x, 2))

    merge['Gap'] = (merge['Costs'] - merge['BKS'].astype('float')) / merge['BKS']
    merge['Gap'] = merge['Gap'].map(lambda x: format(x, ".0%"))
    print(merge)
    output_path = base_path + "jsprit_homberger_%s_BKS_comparison.txt" % size
    merge.to_csv(output_path, index=False, sep="\t")


