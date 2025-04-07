from datamodel import OrderDepth, UserId, TradingState, Order, Trade, Listing
import math
import numpy as np
import collections

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

    outstanding_bids = sorted(list(outstanding.buy_orders.keys()))
    outstanding_asks = sorted(list(outstanding.sell_orders.keys()))

    mid = (outstanding_bids[-1] + outstanding_asks[0]) / 2
    
    max_lag = 4
    mid_prices = [mid]


    if state.traderData:
        mid_prices = [float(p) for p in state.traderData.split(",")[1 - max_lag:]] + [mid]

    # theta4 = [18.40810, 0.16527, 0.14608, 0.27305, 0.40648]
    # theta4 = [18.40810, 0.40648, 0.27305, 0.14608, 0.16527]
    theta4 = [2.02733, 0.33522, 0.26031, 0.20494, 0.19853]

    predicted = mid
    if len(mid_prices) == 4:
        predicted = theta4[0]
        for i in range(1, len(theta4)):
            if i <= len(mid_prices):
                predicted += theta4[i] * mid_prices[-i]

    print(f"Here::0::0::{rr_vol}::{predicted}::0::Here")

    center = round(predicted)


    sell_threshold = center + 1
    buy_threshold = center - 1
    
    short_q = 50 + rr_vol
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

    long_q = rr_vol - 50
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

    updated_data = ",".join(map(str, mid_prices))

    return orders, updated_data

    # product = "KELP"

    # soft_thresh = 40
    # hard_thresh = 45
    # position_limit = 50

    # # Adjust thresholds based on position
    # if abs(rr_vol) >= hard_thresh:
    #     # Hard liquidation mode
    #     if rr_vol > 0:
    #         # Sell aggressively
    #         for k in outstanding_bids[::-1]:
    #             orders.append(Order(product, k, -min(rr_vol, outstanding.buy_orders[k])))
    #             rr_vol -= min(rr_vol, outstanding.buy_orders[k])
    #             if rr_vol <= 0:
    #                 break
    #         if rr_vol > 0:
    #             orders.append(Order(product, outstanding_asks[0], -rr_vol))  # Cross ask
    #     elif rr_vol < 0:
    #         # Buy aggressively
    #         for k in outstanding_asks:
    #             orders.append(Order(product, k, -max(rr_vol, -outstanding.sell_orders[k])))
    #             rr_vol += min(-rr_vol, outstanding.sell_orders[k])
    #             if rr_vol >= 0:
    #                 break
    #         if rr_vol < 0:
    #             orders.append(Order(product, outstanding_bids[-1], -rr_vol))  # Cross bid

    # else:
    #     # Normal market making mode with soft liquidation skew
    #     # Skew prices based on position
    #     buy_skew = 0
    #     sell_skew = 0

    #     if rr_vol >= soft_thresh:
    #         # Soft liquidation: sell more aggressively
    #         sell_skew = -1
    #         buy_skew = -1
    #     elif rr_vol <= -soft_thresh:
    #         # Soft liquidation: buy more aggressively
    #         sell_skew = 1
    #         buy_skew = 1

    #     sell_threshold = center + 1 + sell_skew
    #     buy_threshold = center - 1 + buy_skew

    #     short_q = position_limit + rr_vol
    #     for k in outstanding_bids[::-1]:
    #         if k >= sell_threshold and short_q > 0:
    #             orders.append(Order(product, k, -min(short_q, outstanding.buy_orders[k])))
    #             short_q -= outstanding.buy_orders[k]

    #     if short_q > 0:
    #         orders.append(Order(product,
    #                             max(min(outstanding_asks) - 1, sell_threshold),
    #                             -short_q))

    #     long_q = rr_vol - position_limit
    #     for k in outstanding_asks:
    #         if k <= buy_threshold and long_q < 0:
    #             orders.append(Order(product, k, -max(long_q, outstanding.sell_orders[k])))
    #             long_q -= outstanding.sell_orders[k]

    #     if long_q < 0:
    #         orders.append(Order(product,
    #                             min(max(outstanding_bids) + 1, buy_threshold),
    #                             -long_q))

    # updated_data = ",".join(map(str, mid_prices))
    # return orders, updated_data


class Trader:
    def run(self, state: TradingState):

        orders = {}

        orders["RAINFOREST_RESIN"] = RAINFOREST_RESIN_MM(state)
        orders["KELP"], trader_data = KELP_MM(state)

        return orders, 0, trader_data


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
    traderData = "10.5"

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