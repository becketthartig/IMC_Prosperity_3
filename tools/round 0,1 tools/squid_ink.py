from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
import math
from abc import ABC, abstractmethod
from statistics import NormalDist
import numpy as np

from typing import Any
import json

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()

class BaseMarketMaker(ABC):
    def __init__(self, state, product):
        self.state = state
        self.product = product
        self.orders = []

        self.position = self.state.position.get(self.product, 0)
        self.order_depth = self.state.order_depths[self.product]
        self.outstanding_bids = sorted(list(self.order_depth.buy_orders.keys()))
        self.outstanding_asks = sorted(list(self.order_depth.sell_orders.keys()))
        
        # To be set by subclass
        self.sell_threshold = None
        self.buy_threshold = None

    @abstractmethod
    def compute_thresholds(self):
        """Must set self.sell_threshold and self.buy_threshold"""
        pass

    def make_orders(self):
        self.compute_thresholds()

        short_q = 50 + self.position
        for k in self.outstanding_bids[::-1]:
            if k >= self.sell_threshold and short_q > 0:
                self.orders.append(Order(self.product, k, -min(short_q, self.order_depth.buy_orders[k])))
                short_q -= self.order_depth.buy_orders[k]

        if short_q > 0:
            self.orders.append(Order(self.product, max(min(self.outstanding_asks) - 1, self.sell_threshold + 1), -short_q))

        long_q = self.position - 50
        for k in self.outstanding_asks:
            if k <= self.buy_threshold and long_q < 0:
                self.orders.append(Order(self.product, k, -max(long_q, self.order_depth.sell_orders[k])))
                long_q -= self.order_depth.sell_orders[k]

        if long_q < 0:
            self.orders.append(Order(self.product, min(max(self.outstanding_bids) + 1, self.buy_threshold - 1), -long_q))

        return self.orders


# Stateless function to calulate z-score for a rolling window
def rolling_tick(state_str, new_price, z_score_window, momentum_window, synthetic_mean=False):
    if state_str:
        state = list(map(float, state_str.split(",")))
        prices = state[:-2]
        sum_x = state[-2]
        sum_x2 = state[-1]
    else:
        prices = []
        sum_x = 0
        sum_x2 = 0

    prices.append(new_price)
    sum_x += new_price
    sum_x2 += new_price ** 2

    if len(prices) > z_score_window:
        old_price = prices.pop(0)
        sum_x -= old_price
        sum_x2 -= old_price ** 2

    if len(prices) == z_score_window:
        mean = 0 if synthetic_mean else sum_x / z_score_window
        variance = (sum_x2 - (sum_x ** 2) / z_score_window) / z_score_window
        std = variance ** 0.5 if variance > 0 else 0
        z_score = (new_price - mean) / std if std > 0 else 0
    else:
        z_score = 0

    if len(prices) >= momentum_window:
        momentum = new_price - prices[-momentum_window]
    else:
        momentum = 0

    return ",".join(map(str, prices + [sum_x, sum_x2])), z_score, momentum



def SQUID_INK_MM(state, squid_ink_traderData):

    orders = []
    
    si_vol = state.position.get("SQUID_INK", 0)

    outstanding = state.order_depths["SQUID_INK"]

    outstanding_bids = sorted(list(outstanding.buy_orders.keys()))
    outstanding_asks = sorted(list(outstanding.sell_orders.keys()))

    mid = (outstanding_bids[-1] + outstanding_asks[0]) / 2
    rm = round(mid)
    if mid != rm:
        if si_vol > 0:
            mid = math.floor(mid)
        else:
            mid = rm

            
    z_score_window = 440
    momentum_window = 100

    updated_squid_ink_traderData, z_score, momentum = rolling_tick(squid_ink_traderData, mid, z_score_window, momentum_window)

    center = round(mid)

    sell_threshold = center + 2
    buy_threshold = center - 2

    # higher propensity to sell if z_score > 2
    # higher propensity of buy if z_score < -2
    if z_score > 2:
        sell_threshold -= 3
        buy_threshold -= 3
    elif z_score < -2:
        sell_threshold += 3
        buy_threshold += 3


    short_q = 50 + si_vol
    for k in outstanding_bids[::-1]:
        if k >= sell_threshold and short_q > 0:
            orders.append(Order("SQUID_INK", 
                                k, 
                                -min(short_q, outstanding.buy_orders[k])))
            short_q -= outstanding.buy_orders[k]

    if short_q > 0:
        orders.append(Order("SQUID_INK", 
                            max(min(outstanding_asks) - 1, sell_threshold), 
                            -short_q))

    long_q = si_vol - 50
    for k in outstanding_asks:
        if k <= buy_threshold and long_q < 0:
            orders.append(Order("SQUID_INK", 
                                k, 
                                -max(long_q, outstanding.sell_orders[k])))
            long_q -= outstanding.sell_orders[k]
        
    if long_q < 0:
        orders.append(Order("SQUID_INK", 
                            min(max(outstanding_bids) + 1, buy_threshold), 
                            -long_q))
            
    return orders, updated_squid_ink_traderData



class Trader:

    def run(self, state: TradingState):

        orders = {}

        all_traderData = state.traderData.split(";") if state.traderData else ["", "", "", "", "", "", ""]

        # orders["RAINFOREST_RESIN"] = RainforestResinMM(state).make_orders()
        # orders["KELP"], kelp_traderData = KelpMM(state, all_traderData[0]).make_orders()
        orders["SQUID_INK"], squid_ink_traderData = SQUID_INK_MM(state, all_traderData[1])

        # com_ords_r2, pb1_traderData, pb2_traderData = arb_orders_for_round_2(state, all_traderData[2], all_traderData[3])
        # for k in com_ords_r2:
        #     orders[k] = com_ords_r2[k]

        kelp_traderData = "hahah"
        # squid_ink_traderData = "hahah"
        pb1_traderData = "hahah"
        pb2_traderData = "hahah"

        # com_ords_r3 = volcano_orders(state)
        # for k in com_ords_r3:
        #     orders[k] = com_ords_r3[k]

        ntd = ";".join([kelp_traderData, squid_ink_traderData, pb1_traderData, pb2_traderData])

        logger.flush(state, orders, 0, ntd)


        return orders, 0, ";".join([kelp_traderData, squid_ink_traderData, pb1_traderData, pb2_traderData])


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
