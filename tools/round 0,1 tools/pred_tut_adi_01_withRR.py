from datamodel import OrderDepth, UserId, TradingState, Order, Trade, Listing
import math

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

    # mid = round((outstanding_bids[-1] + outstanding_asks[0]) / 2)

    mid = (outstanding_bids[-1] + outstanding_asks[0]) / 2 # stopped rounding
    
    max_lag = 4
    mid_prices = [mid]
    
    # if state.traderData:
    #     all_data = state.traderData.split(",")
    #     if len(all_data) >= max_lag:
    #         all_data = all_data[-(max_lag - 1):]
    #     mid_prices = list(map(int, all_data)) + [mid]

    if state.traderData:
        all_data = state.traderData.split(",")
        if len(all_data) >= max_lag:
            all_data = all_data[-(max_lag - 1):]
        mid_prices = list(map(float, all_data)) + [mid] # changed to float from int

    theta = [16.18619, -0.02505, 0.00023, 0.00807, -0.00813, 0.00090, 0.04020, 0.04369, 0.06814, 0.11493, 0.11367, 0.25151, 0.38380]

    theta5 = [17.45335, 0.09147, 0.12850, 0.12096, 0.25920, 0.39122]

    theta4 = [18.40810, 0.16527, 0.14608, 0.27305, 0.40648]
    # theta4r = [22.97906, 0.15426, 0.16942, 0.27375, 0.39118]
    # theta4r2 = [15.34356, 0.15184, 0.15857, 0.29516, 0.38680]

    print(mid_prices)

    predicted = mid
    if len(mid_prices) == 4:
        predicted = theta4[0]
        for i in range(1, len(theta4)):
            if i <= len(mid_prices):
                predicted += theta4[i] * mid_prices[-i]

    center = mid
    
    sell_threshold = round(predicted + 1)
    buy_threshold = round(predicted - 1)

    print(f"Here::0::0::{rr_vol}::{predicted}::0::Here")


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

    updated_data = ",".join(map(str, mid_prices[-max_lag:]))
    return orders, updated_data


class Trader:
    def run(self, state: TradingState):

        orders = {}

        # orders["RAINFOREST_RESIN"] = RAINFOREST_RESIN_MM(state)
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
    traderData = ""

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