import pandas as pd
"""
唱凯片区转换为json格式
"""
changkai = pd.read_excel("data/changkai_district.xlsx", dtype=object, names=['id', 'lng', 'lat', 'route', 'series', 'weight'])
changkai['serviceTime'] = 5
cur_res = changkai.iloc[1:]
changkai_district = changkai.iloc[1:][['id', 'lng', 'lat','weight','serviceTime']]
# changkai_district.to_csv("data/inputNodeSingle.csv", index=False)


"""
分析唱凯片区我方结果与苏州环境云结果
"""
import decimal as dc
import numpy as np

dc.getcontext().rounding = dc.ROUND_HALF_UP

cur_res = cur_res[['id', 'route', 'weight', 'lng', 'lat']]
# print(cur_res)
matrix_id = pd.read_csv("data/location_changkai.csv", names=['id', 'lat', 'lng'],
                        dtype={'id': int, 'lat': str, 'lng': str})
# matrix_id[['lat', 'lng']] = matrix_id[['lat', 'lng']].astype(str)
# print(matrix_id)
matrix_id = matrix_id.set_index(['lng', 'lat'])
# print(matrix_id)
matrix = pd.read_csv("data/matrix_changkai.csv", names=['order', 'fromId', 'toId', 'distance', 'time']).set_index(
    ['fromId', 'toId'])
# print(matrix.loc[,:])
depot_id = matrix_id.loc['116.313324330542', '28.1209760513841']['id']

statistic = {}
fromIdList = matrix.reset_index().drop_duplicates().values
for key, value in cur_res.groupby('route'):
    route_name = "路线%s" % key
    distance = 0.;
    weight = 0.;
    time = 0.;
    last_id = np.int64(depot_id)
    for i in range(len(value) - 1):
        cur_lng = str(value.iloc[i]['lng'])
        cur_lat = str(value.iloc[i]['lat'])

        cur_id = np.int64(matrix_id.loc[cur_lng, cur_lat]['id'])

        res = (0, 0)
        if (last_id != cur_id):
            #             if(last_id in fromIdList):
            #                 res = matrix.loc[last_id, cur_id][['distance', 'time']]
            #             else:
            #                 res = matrix.loc[cur_id, last_id][['distance', 'time']]
            try:
                print(type(cur_id), type(last_id))

                print("进入正常处理分支", "cur_id: ", cur_id, "last_id: ", last_id)
                res = matrix.loc[last_id, cur_id][['distance', 'time']]
            except:
                print(type(cur_id), type(last_id))
                print("进入异常处理分支", "cur_id: ", cur_id, "last_id: ", last_id)
                #                 cur_id = int(21863)
                #                 last_id = int(22125)
                #                 print(matrix.loc[cur_id, last_id][['distance', 'time']])
                res = matrix.loc[cur_id, last_id][['distance', 'time']]
        #         print(res)
        distance += res[0]
        time += res[1]
        time += 300  # 秒
        weight += value.iloc[i]['weight']
        last_id = cur_id

    # 回仓
    res = (0, 0)
    if (last_id != depot_id):
        #         if(last_id in fromIdList):
        #             res = matrix.loc[last_id, depot_id][['distance', 'time']]
        #         else:
        #             res = matrix.loc[depot_id, last_id][['distance', 'time']]
        try:
            res = matrix.loc[last_id, depot_id][['distance', 'time']]
        except:
            res = matrix.loc[depot_id, last_id][['distance', 'time']]
    distance += res[0]
    time += res[1]
    statistic[route_name] = {'行驶距离': distance / 1000, '行驶时间': dc.Decimal(time / 60, dc.getcontext()).__round__(0),
                             '垃圾重量': weight}

final_res = pd.DataFrame(statistic)
final_res['总计'] = final_res.sum(axis=1)
final_res
