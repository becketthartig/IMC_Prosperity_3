from datamodel import OrderDepth, UserId, TradingState, Order, Trade, Listing
import math

class Trader:

    def run(self, state: TradingState):

        orders = {"RAINFOREST_RESIN": []}

        rr_vol = state.position.get("RAINFOREST_RESIN", 0)

        outstanding = state.order_depths["RAINFOREST_RESIN"]
        
        outstanding_bids = sorted(list(outstanding.buy_orders.keys())) # trying to buy at
        outstanding_asks = sorted(list(outstanding.sell_orders.keys())) # trying to sell at
        
        sell_threshold = 10001
        buy_threshold = 9999

        short_q = 50 + rr_vol
        for k in outstanding_bids[::-1]:
            if k >= sell_threshold and short_q > 0:
                orders["RAINFOREST_RESIN"].append(Order("RAINFOREST_RESIN", 
                                                        k, 
                                                        -min(short_q, outstanding.buy_orders[k])))
                short_q -= outstanding.buy_orders[k]

        if short_q > 0:
            if short_q < 25:
                orders["RAINFOREST_RESIN"].append(Order("RAINFOREST_RESIN", 
                                                        max(min(outstanding_asks) - 1, sell_threshold + 1), 
                                                        -short_q))
            else:
                orders["RAINFOREST_RESIN"].append(Order("RAINFOREST_RESIN", 
                                                        max(min(outstanding_asks) - 1, sell_threshold + 1), 
                                                        -25))
                orders["RAINFOREST_RESIN"].append(Order("RAINFOREST_RESIN", 
                                                        max(min(outstanding_asks) - 1, sell_threshold), 
                                                        25 - short_q))

        long_q = rr_vol - 50
        for k in outstanding_asks:
            if k <= buy_threshold and long_q < 0:
                orders["RAINFOREST_RESIN"].append(Order("RAINFOREST_RESIN", 
                                                        k, 
                                                        -max(long_q, outstanding.sell_orders[k])))
                long_q -= outstanding.sell_orders[k]
            
        if long_q < 0:
            
            if long_q > -25:
                orders["RAINFOREST_RESIN"].append(Order("RAINFOREST_RESIN", 
                                                        min(max(outstanding_bids) + 1, buy_threshold - 1), 
                                                        -long_q))
            else:
                orders["RAINFOREST_RESIN"].append(Order("RAINFOREST_RESIN", 
                                                        min(max(outstanding_bids) + 1, buy_threshold - 1), 
                                                        25))
                orders["RAINFOREST_RESIN"].append(Order("RAINFOREST_RESIN", 
                                                        min(max(outstanding_bids) + 1, buy_threshold), 
                                                        -25 - long_q))


        print(">>VOL:", rr_vol, state.own_trades, "-----", orders["RAINFOREST_RESIN"], "-----")


        return orders, 0, "Hello"
    

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
    order_depths["KELP"].buy_orders={142: 3, 141: 5},
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