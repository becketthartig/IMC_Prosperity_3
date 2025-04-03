from datamodel import OrderDepth, UserId, TradingState, Order, Trade, Listing, Symbol
import math
import numpy as np
from abc import abstractmethod


class Strategy:
    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit

    @abstractmethod
    def act(self, state: TradingState) -> None:
        raise NotImplementedError()

    def run(self, state: TradingState) -> list[Order]:
        self.orders = []
        self.act(state)
        return self.orders

    def buy(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, quantity))

    def sell(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, -quantity))


class MarketMakingStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)

        # self.window = deque()
        self.window_size = 10

    @abstractmethod
    def get_true_value(state: TradingState) -> int:
        raise NotImplementedError()

    def act(self, state: TradingState) -> None:
        true_value = self.get_true_value(state)

        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position

        max_buy_price = true_value - 1 if position > self.limit * 0.5 else true_value
        min_sell_price = true_value + 1 if position < self.limit * -0.5 else true_value

        for price, volume in sell_orders:
            if to_buy > 0 and price <= max_buy_price:
                quantity = min(to_buy, -volume)
                self.buy(price, quantity)
                to_buy -= quantity

        # if to_buy > 0 and hard_liquidate:
        #     quantity = to_buy // 2
        #     self.buy(true_value, quantity)
        #     to_buy -= quantity

        # if to_buy > 0 and soft_liquidate:
        #     quantity = to_buy // 2
        #     self.buy(true_value - 2, quantity)
        #     to_buy -= quantity

        if to_buy > 0:
            popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
            price = min(max_buy_price, popular_buy_price + 1)
            self.buy(price, to_buy)

        for price, volume in buy_orders:
            if to_sell > 0 and price >= min_sell_price:
                quantity = min(to_sell, volume)
                self.sell(price, quantity)
                to_sell -= quantity

        # if to_sell > 0 and hard_liquidate:
        #     quantity = to_sell // 2
        #     self.sell(true_value, quantity)
        #     to_sell -= quantity

        # if to_sell > 0 and soft_liquidate:
        #     quantity = to_sell // 2
        #     self.sell(true_value + 2, quantity)
        #     to_sell -= quantity

        if to_sell > 0:
            popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
            price = max(min_sell_price, popular_sell_price - 1)
            self.sell(price, to_sell)

class AmethystsStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> int:
        return 10_000

class StarfruitStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> int:
        order_depth = state.order_depths[self.symbol]

        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
        popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]

        return round((popular_buy_price + popular_sell_price) / 2)

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
    center = round((outstanding_bids[-1] + outstanding_asks[0]) / 2)

    kept_data = [str(center)]
    fkept_data = [center]

    old_slope = 0
    signal_tracker = 0
    if state.traderData:
        all_data = state.traderData.split(";")
        old_slope = float(all_data[1])
        signal_tracker = float(all_data[2])
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

    print(f"Here::{signal_tracker}::{roc}::{rr_vol}::{center}::{slope}::Here")



    # sell_threshold = center + 1
    # buy_threshold = center - 1

    spread = 2 #+ abs(slope) * 20  # Wider spread if slope is high
    sell_threshold = center + spread // 2
    buy_threshold = center - spread // 2

    if slope > 0.055:  # Uptrend
        buy_threshold -= 1  # Buy more aggressively
        sell_threshold += 2  # Sell more conservatively
    elif slope < -0.055:  # Downtrend
        buy_threshold -= 2  # Buy more conservatively
        sell_threshold += 1 

    if slope > 0 and rr_vol < 0:  # Market is rising, you're net short -> start covering
        buy_threshold += 1
    elif slope < 0 and rr_vol > 0:  # Market is falling, you're net long -> sell quicker
        sell_threshold -= 1

    max_pos = 50


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

    
    return orders, f"{",".join(kept_data)};{slope};{signal_tracker}"


class Trader:

    def run(self, state: TradingState):

        orders = {}


        ks = StarfruitStrategy("KELP", 50)

        # orders["RAINFOREST_RESIN"] = RAINFOREST_RESIN_MM(state)
        
        # orders["KELP"], kd = KELP_MM(state)
        
        orders["KELP"] = ks.run(state)

        # return orders, 0, kd
        return orders, 0, "j"
    


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
    traderData = "50,60,60,70,80,90,100,100,110,120;4;1"

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