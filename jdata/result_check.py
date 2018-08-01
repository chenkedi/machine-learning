import pandas as pd
import decimal as dc

def timeToMinute(time):
    times = time.split(":")
    return int(times[0]) * 60 + int(times[1])

def minuteToTime(minute):
    return "%s:%s" % (minute // 60, minute % 60)

dc.getcontext().rounding = dc.ROUND_HALF_UP

def checkDistanceCost(row):
    route = row["dist_seq"].split(';')
    vehicleType = row["vehicle_type"]
    distanceCost = typeOneDistanceCost
    if(vehicleType == 2):
        distanceCost = typeTwoDistanceCost
    totalDistance = 0
    for i in range(len(route) - 1):
        fromNode = int(route[i])
        toNode = int(route[i + 1])
        totalDistance += matrix.ix[fromNode, toNode]["distance"]

    return totalDistance == row["distance"], \
           dc.Decimal(totalDistance * distanceCost, dc.getcontext()).__round__(2) == dc.Decimal(row["trans_cost"], dc.getcontext()).__round__(2)

def checkWaitTimeCost(row):
    route = row["dist_seq"].split(';')
    arrTime = timeToMinute(row["distribute_lea_tm"])
    waitTime = 0
    for i in range(len(route) - 1):
        fromNode = int(route[i])
        toNode = int(route[i + 1])
        arrTime += matrix.ix[fromNode, toNode]["spend_tm"]
        if(toNode > 0 and toNode <= 1000):
            start_tw = timeToMinute(nodes.ix[toNode]["first_receive_tm"])
            if(arrTime < start_tw):
                 waitTime += (start_tw - arrTime)
                 arrTime = start_tw

            if(arrTime > timeToMinute(nodes.ix[toNode]["last_receive_tm"])):
                 print("Timewindow break in line %s and activity %s,"
                       " arrTime is %s, last receive tm is %s", (row["trans_code"], toNode, arrTime, timeToMinute(nodes.ix[toNode]["last_receive_tm"])))
        arrTime += serviceTime

    return dc.Decimal(waitTime * waitCost, dc.getcontext()).__round__(2) == dc.Decimal(row["wait_cost"], dc.getcontext()).__round__(2)



def totalCostCheck(row):
    return dc.Decimal(row["trans_cost"] + row["charge_cost"] + row["wait_cost"] + row["fixed_use_cost"], dc.getcontext()).__round__(2) \
           == dc.Decimal(row["total_cost"], dc.getcontext()).__round__(2)

dir = "try_2"
path = "raw_data/results/%s/Result.csv" % dir

chargeCost = 50
typeOneDistanceCost = 0.012
typeTwoDistanceCost = 0.014
waitCost = 0.4
serviceTime = 30
chargeTime = 30

res = pd.read_csv(path)
nodes = pd.read_json("output_data/input_node.json").set_index("ID")
# print(nodes)

matrix = pd.read_csv("raw_data/input_distance-time.txt", index_col=[1, 2])

# for i in range(0, len(matrix)):
#     distance_matrix["%s->%s" % (matrix.iloc[i]["from_node"], matrix.iloc[i]["to_node"])] = matrix.iloc[i]["distance"]
#     time_matrix["%s->%s" % (matrix.iloc[i]["from_node"], matrix.iloc[i]["to_node"])] = matrix.iloc[i]["spend_tm"]

res["distance_check"] = res.apply(lambda row: checkDistanceCost(row)[0], axis=1)
res["transport_check"] = res.apply(lambda row: checkDistanceCost(row)[1], axis=1)
res["totalCost_check"] = res.apply(lambda row: totalCostCheck(row), axis=1)
res["waitCost_check"] = res.apply(lambda row: checkWaitTimeCost(row), axis=1)

print("Distance error: %s" % len(res[res["distance_check"] == False]))
print("transport cost error %s" % len(res[res["transport_check"] == False]))
print("totalCost error: %s" % len(res[res["totalCost_check"] == False]))
print("waitCost error: %s" % len(res[res["waitCost_check"] == False]))


