from datamodel import OrderDepth, UserId, TradingState, Order, Trade, Listing
import math


def RAINFOREST_RESIN_MM(state):
    orders = []
    symbol = "RAINFOREST_RESIN"
    
    rr_vol = state.position.get(symbol, 0)
    outstanding = state.order_depths[symbol]

    outstanding_bids = sorted(outstanding.buy_orders.keys())
    outstanding_asks = sorted(outstanding.sell_orders.keys())
    
    if not outstanding_bids or not outstanding_asks:
        return [], state.traderData  # Not enough info

    mid = (outstanding_bids[-1] + outstanding_asks[0]) / 2
    max_lag = 5
    mid_prices = [mid]

    if state.traderData:
        data = state.traderData.split(";")
        resin_data = next((d for d in data if d.startswith("RR:")), "")
        if resin_data:
            rr_vals = resin_data.replace("RR:", "").split(",")
            rr_vals = list(map(float, rr_vals[-(max_lag - 1):]))
            mid_prices = rr_vals + [mid]

    # === Linear Regression Prediction ===
    predicted = mid
    if len(mid_prices) >= 3:
        X = list(range(len(mid_prices)))
        x_mean = sum(X) / len(X)
        y_mean = sum(mid_prices) / len(mid_prices)
        numer = sum((X[i] - x_mean) * (mid_prices[i] - y_mean) for i in range(len(X)))
        denom = sum((X[i] - x_mean) ** 2 for i in range(len(X)))
        slope = numer / denom if denom != 0 else 0
        intercept = y_mean - slope * x_mean
        predicted = slope * len(X) + intercept

    alpha = predicted - mid

    # === Aggressive mean-reversion thresholds ===
    base_buy_offset = 2.0
    base_sell_offset = 2.0

    buy_threshold = mid - (base_buy_offset - 0.5 * alpha)
    sell_threshold = mid + (base_sell_offset + 0.5 * alpha)

    # === Position-based adjustment (bias thresholds) ===
    # Positive rr_vol = long → raise buy threshold (less aggressive), raise sell threshold (more eager to unload)
    buy_threshold += 0.02 * rr_vol
    sell_threshold += 0.02 * rr_vol

    # Round to nearest tick (1 for now, adjust if needed)
    buy_threshold = round(buy_threshold)
    sell_threshold = round(sell_threshold)

    # === SELL logic ===
    short_q = 50 + rr_vol
    for k in sorted(outstanding.buy_orders.keys(), reverse=True):
        if k >= sell_threshold and short_q > 0:
            qty = min(short_q, outstanding.buy_orders[k])
            orders.append(Order(symbol, k, -qty))
            short_q -= qty

    if short_q > 0 and outstanding_asks:
        fallback_price = max(min(outstanding_asks) - 1, sell_threshold)
        orders.append(Order(symbol, fallback_price, -short_q))

    # === BUY logic ===
    long_q = rr_vol - 50
    for k in sorted(outstanding.sell_orders.keys()):
        if k <= buy_threshold and long_q < 0:
            qty = max(long_q, outstanding.sell_orders[k])
            orders.append(Order(symbol, k, -qty))
            long_q -= outstanding.sell_orders[k]

    if long_q < 0 and outstanding_bids:
        fallback_price = min(max(outstanding_bids) + 1, buy_threshold)
        orders.append(Order(symbol, fallback_price, -long_q))

    updated_data = f"RR:{','.join(map(str, mid_prices[-max_lag:]))}"
    return orders, updated_data


class Trader:
    def run(self, state: TradingState):
        orders = {}
        resin_orders, resin_data = RAINFOREST_RESIN_MM(state)
        orders["RAINFOREST_RESIN"] = resin_orders
        trader_data = resin_data
        return orders, 0, trader_data


# Optional test harness
if __name__ == "__main__":
    timestamp = 1100

    listings = {
        "RAINFOREST_RESIN": Listing(
            symbol="RAINFOREST_RESIN", 
            product="RAINFOREST_RESIN", 
            denomination="SEASHELLS"
        ),
    }

    order_depths = {
        "RAINFOREST_RESIN": OrderDepth(),
    }

    order_depths["RAINFOREST_RESIN"].buy_orders = {10: 7, 9: 5}
    order_depths["RAINFOREST_RESIN"].sell_orders = {12: -5, 13: -3}

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
        ]
    }

    market_trades = {
        "RAINFOREST_RESIN": []
    }

    position = {
        "RAINFOREST_RESIN": 10
    }

    observations = {}
    traderData = "RR:10.3,10.4,10.5"

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