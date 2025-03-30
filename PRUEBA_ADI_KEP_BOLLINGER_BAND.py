from datamodel import OrderDepth, UserId, TradingState, Order, Trade, Listing
import math
import pandas as pd

rolling_data = pd.DataFrame(columns=["timestamp", "mid_price"])


# Making a df of timestamps and midprices
def update_rolling_data(rolling_data, timestamp, mid_price, window=20):

    new_row = {"timestamp": timestamp, "mid_price": mid_price}
    rolling_data = pd.concat([rolling_data, pd.DataFrame([new_row])], ignore_index=True)
    
    if len(rolling_data) > window:
        rolling_data = rolling_data.iloc[-window:]
    
    return rolling_data

def calculate_bollinger_bands(df, column_name="mid_price", window=20):
    df["Rolling Mean"] = df[column_name].rolling(window=window).mean()
    df["Rolling Std"] = df[column_name].rolling(window=window).std()
    df["Upper Band"] = df["Rolling Mean"] + (2 * df["Rolling Std"])
    df["Lower Band"] = df["Rolling Mean"] - (2 * df["Rolling Std"])
    return df

def generate_signals(df):
    if len(df) < 1 or pd.isna(df.iloc[-1]["Upper Band"]) or pd.isna(df.iloc[-1]["Lower Band"]):
        return "HOLD"
    latest_row = df.iloc[-1]
    
    if latest_row["mid_price"] < latest_row["Lower Band"]:
        return "BUY"
    elif latest_row["mid_price"] > latest_row["Upper Band"]:
        return "SELL"
    
    return "HOLD"


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
    global rolling_data

    orders = []

    kelp_vol = state.position.get("KELP", 0)

    outstanding = state.order_depths["KELP"]

    outstanding_bids = sorted(list(outstanding.buy_orders.keys()))
    outstanding_asks = sorted(list(outstanding.sell_orders.keys()))

    if outstanding_bids and outstanding_asks:
        best_bid = max(outstanding_bids)
        best_ask = min(outstanding_asks)
        mid_price = (best_bid + best_ask) / 2

        rolling_data = update_rolling_data(rolling_data, state.timestamp, mid_price)

        rolling_data = calculate_bollinger_bands(rolling_data)

        signal = generate_signals(rolling_data)

        max_position = 100
        order_size = 10

        if signal == "BUY" and kelp_vol < max_position:
            quantity_to_buy = min(order_size, max_position - kelp_vol)
            orders.append(Order("KELP", best_ask, quantity_to_buy))

        elif signal == "SELL" and kelp_vol > -max_position:
            quantity_to_sell = min(order_size, kelp_vol + max_position)
            orders.append(Order("KELP", best_bid, -quantity_to_sell))

    return orders, rolling_data.tail(1)


class Trader:

    def __init__(self):
        self.rolling_data = pd.DataFrame(columns=["timestamp", "mid_price"])

    def run(self, state: TradingState):

        orders = {}

        # orders["RAINFOREST_RESIN"] = RAINFOREST_RESIN_MM(state)

        kelp_orders, latest_bollinger_info = KELP_MM(state)
        orders["KELP"] = kelp_orders

        return orders, 0, f"Latest Bollinger Info: {latest_bollinger_info.to_dict()}"



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