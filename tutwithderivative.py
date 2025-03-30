from datamodel import OrderDepth, UserId, TradingState, Order, Trade, Listing
import math
import numpy as np

def RAINFOREST_RESIN_MM(state):

    orders = []
    
    rr_vol = state.position.get("RAINFOREST_RESIN", 0)
    outstanding = state.order_depths["RAINFOREST_RESIN"]

    outstanding_bids = sorted(list(outstanding.buy_orders.keys())) # trying to buy at
    outstanding_asks = sorted(list(outstanding.sell_orders.keys())) # trying to sell at
    
    sell_threshold = 10001
    buy_threshold = 9999

    short_q = 50 + rr_vol
    for k in outstanding_bids[::-1]:
        if k >= sell_threshold and short_q > 0:
            orders.append(Order("RAINFOREST_RESIN", 
                                k, 
                                -min(short_q, outstanding.buy_orders[k])))
            short_q -= outstanding.buy_orders[k]

    if short_q > 0:
        orders.append(Order("RAINFOREST_RESIN", 
                            max(min(outstanding_asks) - 1, sell_threshold + 1), 
                            -short_q))

    long_q = rr_vol - 50
    for k in outstanding_asks:
        if k <= buy_threshold and long_q < 0:
            orders.append(Order("RAINFOREST_RESIN", 
                                k, 
                                -max(long_q, outstanding.sell_orders[k])))
            long_q -= outstanding.sell_orders[k]
        
    if long_q < 0:
        orders.append(Order("RAINFOREST_RESIN", 
                            min(max(outstanding_bids) + 1, buy_threshold - 1), 
                            -long_q))
            
    return orders



def KELP_MM(state):

    orders = []

    rr_vol = state.position.get("KELP", 0)
    outstanding = state.order_depths["KELP"]

    outstanding_bids = sorted(list(outstanding.buy_orders.keys())) # trying to buy at
    outstanding_asks = sorted(list(outstanding.sell_orders.keys())) # trying to sell at

    # center = round((outstanding_bids[-1] + outstanding_asks[0]) / 2)
    center = (outstanding_bids[-1] + outstanding_asks[0]) // 2 

    kept_data = [str(center)]
    fkept_data = [center]

    old_slope = 0
    if state.traderData:
        all_data = state.traderData.split(";")
        old_slope = float(all_data[1])
        kept_data = all_data[0].split(",")
        if len(kept_data) >= 20:
            del kept_data[0]
        kept_data.append(str(center))
        fkept_data = [float(point) for point in kept_data]

    slope = 0
    if len(fkept_data) > 1:
        steps = np.arange(len(fkept_data))
        slope = np.polyfit(steps, fkept_data, 1)[0]

    roc = slope - old_slope

    print(f"Here::{roc}::{rr_vol}::{center}::{slope}::Here")



    sell_threshold = center
    buy_threshold = center
    # if slope > 0.05:
    #     sell_threshold += 2
    # elif slope < -0.05:
    #     buy_threshold -= 2
    # else:
    #     sell_threshold += 1
    #     buy_threshold -= 1

    max_pos = 50
    if roc >= 0.0125 and slope >= 0:
        sell_threshold += 3
        buy_threshold += 1
    elif roc <= -0.0125  and slope <= 0:
        buy_threshold -= 2
    elif slope >= 0.055:
        sell_threshold += 3
        buy_threshold += 1
    elif slope <= -0.055:
        buy_threshold -= 2
    else:
        max_pos = 25
        sell_threshold += 1
        buy_threshold -= 1

    # if slope > 0:
    #     sell_threshold += 1
    #     buy_threshold += 1


    sell_threshold = int(round(sell_threshold))  # Ensure integer price
    buy_threshold = int(round(buy_threshold))  # Ensure integer price

    print("NEXT:", center, buy_threshold, sell_threshold)

    print("NNEXT:", outstanding_bids, outstanding_asks)



    short_q = max_pos + rr_vol
    for k in outstanding_bids[::-1]:
        if k >= sell_threshold and short_q > 0:
            orders.append(Order("KELP", 
                                k, 
                                -min(short_q, outstanding.buy_orders[k])))
            short_q -= outstanding.buy_orders[k]

    if short_q > 0:
        orders.append(Order("KELP", 
                            max(min(outstanding_asks) - 1, sell_threshold), 
                            -short_q))

    long_q = rr_vol - max_pos
    for k in outstanding_asks:
        if k <= buy_threshold and long_q < 0:
            orders.append(Order("KELP", 
                                k, 
                                -max(long_q, outstanding.sell_orders[k])))
            long_q -= outstanding.sell_orders[k]
        
    if long_q < 0:
        orders.append(Order("KELP", 
                            min(max(outstanding_bids) + 1, buy_threshold), 
                            -long_q))

    # spread = max(1, min(3, 3 - abs(slope)))


    # **Shift buy/sell thresholds based on trend**
    # sell_threshold = center + (2 if slope < 0 else 1)
    # buy_threshold = center - (0 if slope < 0 else 1)

    print("ORDERS:", orders)

    
    return orders, f"{",".join(kept_data)};{slope}"


class Trader:

    def run(self, state: TradingState):

        orders = {}


        # orders["RAINFOREST_RESIN"] = RAINFOREST_RESIN_MM(state)
        
        orders["KELP"], kd = KELP_MM(state)
        

        return orders, 0, kd
    


if __name__ == "__main__":
    timestamp = 1100

    listings = {
        "RAINFOREST_RESIN": Listing(
            symbol="RAINFOREST_RESIN", 
            product="RAINFOREST_RESIN", 
            denomination="SEASHELLS"
        ),
        "KELP": Listing(
            symbol="KELP", 
            product="KELP", 
            denomination="SEASHELLS"
        ),
    }

    order_depths = {
        "RAINFOREST_RESIN": OrderDepth(),
        "KELP": OrderDepth(),	
    }

    order_depths["RAINFOREST_RESIN"].buy_orders={10: 7, 9: 5}
    order_depths["RAINFOREST_RESIN"].sell_orders={12: -5, 13: -3}
    order_depths["KELP"].buy_orders={142: 3, 141: 5}
    order_depths["KELP"].sell_orders={144: -5, 145: -8}
    

    own_trades = {
        "RAINFOREST_RESIN": [
            Trade(
                symbol="RAINFOREST_RESIN",
                price=11,
                quantity=4,
                buyer="SUBMISSION",
                seller="",
                timestamp=1000
            ),
            Trade(
                symbol="RAINFOREST_RESIN",
                price=12,
                quantity=3,
                buyer="SUBMISSION",
                seller="",
                timestamp=1000
            )
        ],
        "KELP": [
            Trade(
                symbol="KELP",
                price=143,
                quantity=2,
                buyer="",
                seller="SUBMISSION",
                timestamp=1000
            ),
        ]
    }

    market_trades = {
        "RAINFOREST_RESIN": [],
        "KELP": []
    }

    position = {
        "RAINFOREST_RESIN": 10,
        "KELP": -7
    }

    observations = {}
    traderData = "50,60,60,70,80,90,100,100,110,120;4"

    s = TradingState(
        traderData,
        timestamp,
        listings,
        order_depths,
        own_trades,
        market_trades,
        position,
        observations
    )

    T = Trader()
    print(T.run(s))