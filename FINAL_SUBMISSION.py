from datamodel import Order, TradingState
import math
from abc import ABC, abstractmethod
from statistics import NormalDist
import pandas as pd


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
            self.orders.append(Order(self.product,
                                     max(min(self.outstanding_asks) - 1, self.sell_threshold + 1),
                                     -short_q))

        long_q = self.position - 50
        for k in self.outstanding_asks:
            if k <= self.buy_threshold and long_q < 0:
                self.orders.append(Order(self.product, k, -max(long_q, self.order_depth.sell_orders[k])))
                long_q -= self.order_depth.sell_orders[k]

        if long_q < 0:
            self.orders.append(Order(self.product,
                                     min(max(self.outstanding_bids) + 1, self.buy_threshold - 1),
                                     -long_q))

        return self.orders

class RainforestResinMM(BaseMarketMaker):
    def __init__(self, state):
        super().__init__(state, "RAINFOREST_RESIN")

    def compute_thresholds(self):
        self.sell_threshold = 10001
        self.buy_threshold = 9999

class KelpMM(BaseMarketMaker):
    def __init__(self, state, kelp_traderData):
        super().__init__(state, "KELP")
        self.kelp_traderData = kelp_traderData
        self.mid_prices = []

    def compute_thresholds(self):
        mid = (self.outstanding_bids[-1] + self.outstanding_asks[0]) / 2

        max_lag = 4
        if self.kelp_traderData:
            self.mid_prices = [float(p) for p in self.kelp_traderData.split(",")[1 - max_lag:]] + [mid]
        else:
            self.mid_prices = [mid]

        theta4 = (2.02733, 0.33522, 0.26031, 0.20494, 0.19853)

        # theta44 = (1.40546, 0.32203, 0.26097, 0.20511, 0.21121)
        # theta4 = (1.75335, 0.32513, 0.26138, 0.20530, 0.20733)

        predicted = theta4[0]
        for i in range(1, len(theta4)):
            if i <= len(self.mid_prices):
                predicted += theta4[i] * self.mid_prices[-i]

        center = round(predicted)
        self.sell_threshold = center + 1
        self.buy_threshold = center - 1

    def make_orders(self):
        orders = super().make_orders()
        return orders, ",".join(map(str, self.mid_prices))


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

def ORDER_ENGINE(product, short_q, long_q, outstanding_bids, outstanding_asks, sell_threshold, buy_threshold, outstanding):

    orders = []

    for k in outstanding_bids[::-1]:
        if k >= sell_threshold and short_q > 0:
            orders.append(Order(product, 
                                k, 
                                -min(short_q, outstanding.buy_orders[k])))
            short_q -= outstanding.buy_orders[k]

    if short_q > 0:
        mm = 0
        if outstanding_asks:
            mm = min(outstanding_asks)
        else:
            mm = max(outstanding_bids)
        orders.append(Order(product, 
                            max(mm - 1, sell_threshold), # +1 ?
                            -short_q))

    for k in outstanding_asks:
        if k <= buy_threshold and long_q < 0:
            orders.append(Order(product, 
                                k, 
                                -max(long_q, outstanding.sell_orders[k])))
            long_q -= outstanding.sell_orders[k]
        
    if long_q < 0:
        mm = 0
        if outstanding_bids:
            mm = max(outstanding_bids)
        else:
            mm = min(outstanding_asks)
        orders.append(Order(product, 
                            min(mm + 1, buy_threshold), # -1 ?
                            -long_q))
        
    return orders


def arb_orders_for_round_2(state, pb1_traderData, pb2_traderData):
    commodities = ("PICNIC_BASKET1", "PICNIC_BASKET2", "CROISSANTS", "JAMS", "DJEMBES")

    outstanding_bids = {}
    outstanding_asks = {}
    mids = {}
    vols = {}
    for c in commodities:
        outstanding_bids[c] = sorted(state.order_depths[c].buy_orders.keys())
        outstanding_asks[c] = sorted(state.order_depths[c].sell_orders.keys())
        mids[c] = (outstanding_bids[c][-1] + outstanding_asks[c][0]) / 2
        vols[c] = state.position.get(c, 0)

    pb1_synthetic = 6 * mids["CROISSANTS"] + 3 * mids["JAMS"] + mids["DJEMBES"]
    pb2_synthetic = 4 * mids["CROISSANTS"] + 2 * mids["JAMS"]

    premium_pb1 = mids["PICNIC_BASKET1"] - pb1_synthetic
    premium_pb2 = mids["PICNIC_BASKET2"] - pb2_synthetic

    updated_pb1_traderData, zpb1, mpb1 = rolling_tick(pb1_traderData, premium_pb1, 40, 25, True)
    updated_pb2_traderData, zpb2, mpb2 = rolling_tick(pb2_traderData, premium_pb2, 40, 25, True)

    ENGINE_LIMITS = {"PICNIC_BASKET1": 60, "PICNIC_BASKET2": 100, "CROISSANTS": 250, "JAMS": 350, "DJEMBES": 60}

    commodity_orders = {}
    short_qs = {k: 0 for k in commodities}
    long_qs = {k: 0 for k in commodities}

    base_qty = 10

    def z_scaled_qty(z, limit):
        return int(min(limit, base_qty * (abs(z) / 10)))
    

    if (vols["PICNIC_BASKET1"] < 0 and zpb1 < 0) or (vols["PICNIC_BASKET1"] > 0 and zpb1 > 0):
        if vols["PICNIC_BASKET1"] > 0:
            short_qs["PICNIC_BASKET1"] = vols["PICNIC_BASKET1"]
        elif vols["PICNIC_BASKET1"] < 0:
            long_qs["PICNIC_BASKET1"] = vols["PICNIC_BASKET1"]

        if vols["CROISSANTS"] > 0:
            short_qs["CROISSANTS"] = vols["CROISSANTS"]
        elif vols["CROISSANTS"] < 0:
            long_qs["CROISSANTS"] = vols["CROISSANTS"]

        if vols["JAMS"] > 0:
            short_qs["JAMS"] = vols["JAMS"]
        elif vols["JAMS"] < 0:
            long_qs["JAMS"] = vols["JAMS"]

        if vols["DJEMBES"] > 0:
            short_qs["DJEMBES"] = vols["DJEMBES"]
        elif vols["DJEMBES"] < 0:
            long_qs["DJEMBES"] = vols["DJEMBES"]

    elif zpb1 > 10 and mpb1 < 2:
        qty = z_scaled_qty(zpb1, ENGINE_LIMITS["PICNIC_BASKET1"] + vols["PICNIC_BASKET1"])
        short_qs["PICNIC_BASKET1"] += qty
        long_qs["CROISSANTS"] -= 6 * qty
        long_qs["JAMS"] -= 3 * qty
        long_qs["DJEMBES"] -= qty

    elif zpb1 < -10 and mpb1 > -2:
        qty = z_scaled_qty(zpb1, ENGINE_LIMITS["PICNIC_BASKET1"] - vols["PICNIC_BASKET1"])
        long_qs["PICNIC_BASKET1"] -= qty
        short_qs["CROISSANTS"] += 6 * qty
        short_qs["JAMS"] += 3 * qty
        short_qs["DJEMBES"] += qty


    if (vols["PICNIC_BASKET2"] < 0 and zpb2 < 0) or (vols["PICNIC_BASKET2"] > 0 and zpb2 > 0):
        if vols["PICNIC_BASKET2"] > 0:
            short_qs["PICNIC_BASKET2"] = vols["PICNIC_BASKET2"]
        elif vols["PICNIC_BASKET2"] < 0:
            long_qs["PICNIC_BASKET2"] = vols["PICNIC_BASKET2"]

        if vols["CROISSANTS"] > 0:
            short_qs["CROISSANTS"] = vols["CROISSANTS"]
        elif vols["CROISSANTS"] < 0:
            long_qs["CROISSANTS"] = vols["CROISSANTS"]

        if vols["JAMS"] > 0:
            short_qs["JAMS"] = vols["JAMS"]
        elif vols["JAMS"] < 0:
            long_qs["JAMS"] = vols["JAMS"]

    elif zpb2 > 10 and mpb2 < 2:
        qty = z_scaled_qty(zpb2, ENGINE_LIMITS["PICNIC_BASKET2"] + vols["PICNIC_BASKET2"])
        short_qs["PICNIC_BASKET2"] += qty
        long_qs["CROISSANTS"] -= 4 * qty
        long_qs["JAMS"] -= 2 * qty

    elif zpb2 < -10 and mpb2 > -2:
        qty = z_scaled_qty(zpb2, ENGINE_LIMITS["PICNIC_BASKET2"] - vols["PICNIC_BASKET2"])
        long_qs["PICNIC_BASKET2"] -= qty
        short_qs["CROISSANTS"] += 4 * qty
        short_qs["JAMS"] += 2 * qty

    for c in commodities:

        short_qty = min(short_qs[c], ENGINE_LIMITS[c] + vols[c]) if short_qs[c] > 0 else 0
        long_qty = max(long_qs[c], -(ENGINE_LIMITS[c] - vols[c])) if long_qs[c] < 0 else 0

        if c != "JAMS" and c != "DJEMBES" and c != "CROISSANTS":
            commodity_orders[c] = ORDER_ENGINE(
                c,
                short_qty,
                long_qty,
                outstanding_bids[c],
                outstanding_asks[c],
                round(mids[c]) if vols[c] > 0 else math.floor(mids[c]),
                round(mids[c]) if vols[c] > 0 else math.floor(mids[c]),
                state.order_depths[c],
            )

    return commodity_orders, updated_pb1_traderData, updated_pb2_traderData


class BLACK_SCHOLES_CALC():

    def __init__(self, spot_price, days_to_mature, risk_free_rate=0):
        self.S = spot_price
        self.t = days_to_mature / 365
        self.r = risk_free_rate
        self.N = NormalDist().cdf

    def call_price(self, K, vol):
        d1 = (math.log(self.S / K) + (self.r + vol ** 2 / 2) * self.t) / (vol * math.sqrt(self.t))
        d2 = d1 - (vol * math.sqrt(self.t))
        return self.S * self.N(d1) - K * math.exp(-self.r * self.t) * self.N(d2) 
    
    def implied_vol(self, K, C, precision_limit=1e-7, iter_limit=100):
        low = 1e-7
        high = 2
        mid = 0
        for _ in range(iter_limit):
            mid = (low + high) / 2
            price = self.call_price(K, mid)
            if abs(price - C) / max(C, 1e-7) < precision_limit:
                return mid
            elif price < C:
                low = mid
            else:
                high = mid
        return mid 
    
    def delta(self, K, vol):
        d1 = (math.log(self.S / K) + (self.r + vol ** 2 / 2) * self.t) / (vol * math.sqrt(self.t))
        return self.N(d1)
    
    def gamma(self, K, vol):
        d1 = (math.log(self.S / K) + (self.r + vol ** 2 / 2) * self.t) / (vol * math.sqrt(self.t))
        Npd1 = math.exp(-(d1 ** 2) / 2) / math.sqrt(2 * math.pi)
        return Npd1 / (self.S * vol * math.sqrt(self.t))
    
    def vega(self, K, vol):
        d1 = (math.log(self.S / K) + (self.r + vol ** 2 / 2) * self.t) / (vol * math.sqrt(self.t))
        Npd1 = math.exp(-(d1 ** 2) / 2) / math.sqrt(2 * math.pi)
        return (self.S * math.sqrt(self.t) * Npd1) / 100
    

def optimize_DG(
    total_delta,
    total_gamma,
    deltas,
    gammas,
    weights,
    threshold,
    max_coeffs,
    min_coeff=0,
    delta_tolerance=1
):
    keys = list(deltas.keys())
    sign = -1 if total_delta > 0 else 1

    usable_keys = [
        k for k in keys
        if (total_delta < 0 and weights[k] <= threshold)
        or (total_delta > 0 and weights[k] >= -threshold)
    ]
    
    best = None
    best_within_tolerance = None
    best_gamma_diff = float("inf")
    best_delta_error = float("inf")

    for i in range(len(usable_keys)):
        for j in range(i, len(usable_keys)):
            k1, k2 = usable_keys[i], usable_keys[j]
            d1, d2 = deltas[k1], deltas[k2]
            g1, g2 = gammas[k1], gammas[k2]
            m1, m2 = max_coeffs[k1], max_coeffs[k2]

            for c1 in range(min(min_coeff, m1), m1 + 1):
                for c2 in range(min(min_coeff, m2), m2 + 1):
                    c1_signed = -sign * c1
                    c2_signed = -sign * c2

                    delta_sum = c1_signed * d1 + c2_signed * d2
                    delta_error = abs(delta_sum - total_delta)

                    gamma_sum = c1_signed * g1 + c2_signed * g2
                    gamma_diff = abs(gamma_sum - total_gamma)

                    candidate = {
                        'keys': (k1, k2),
                        'coeffs': (-c1_signed, -c2_signed),
                        'delta_sum': -delta_sum,
                        'gamma_sum': -gamma_sum,
                        'delta_diff': delta_error,
                        'gamma_diff': gamma_diff
                    }

                    if delta_error <= delta_tolerance:
                        if gamma_diff < best_gamma_diff:
                            best_gamma_diff = gamma_diff
                            best_within_tolerance = candidate
                    else:
                        if (delta_error < best_delta_error) or (
                            delta_error == best_delta_error and gamma_diff < best_gamma_diff):
                            best_delta_error = delta_error
                            best_gamma_diff = gamma_diff
                            best = candidate

    return best_within_tolerance if best_within_tolerance else best
    

def volcano_orders(state):

    underlying = "VOLCANIC_ROCK"
    calls = ["VOLCANIC_ROCK_VOUCHER_9500", "VOLCANIC_ROCK_VOUCHER_9750", "VOLCANIC_ROCK_VOUCHER_10000", "VOLCANIC_ROCK_VOUCHER_10250", "VOLCANIC_ROCK_VOUCHER_10500"]

    outstanding_bids, outstanding_asks, mids, posns =  {}, {}, {}, {}
    for c in calls + [underlying]:
        outstanding_bids[c] = sorted(state.order_depths[c].buy_orders.keys())
        outstanding_asks[c] = sorted(state.order_depths[c].sell_orders.keys())
        if outstanding_bids[c]:
            if outstanding_asks[c]:
                mids[c] = (outstanding_bids[c][-1] + outstanding_asks[c][0]) / 2
            else:
                mids[c] = outstanding_bids[c][-1]
        elif outstanding_asks[c]:
            mids[c] = outstanding_asks[c][0]

        posns[c] = state.position.get(c, 0)

    #### ****** 7000000 number MUST be changed for final submission ****** ####
    bsmodel = BLACK_SCHOLES_CALC(mids[underlying], (3000000 - state.timestamp) / 1000000)
    IV_fit = (0.23729, 0.00294, 0.14920)

    max_stock_pos = 400
    max_call_pos = 200
    vol = 0.151
    misprice_threshold = 0.002
    delta_hedge_threshold = 10
    
    mispriced_vols, all_mispricings = {}, {underlying: 0}
    for c in calls:
        strike = int(c.split("_")[-1])
        iv = bsmodel.implied_vol(strike, mids[c])
        moneyness = math.log(strike / mids[underlying]) / math.sqrt(bsmodel.t)
        ev = IV_fit[0] * moneyness ** 2 + IV_fit[1] * moneyness + IV_fit[2]
        misprice = iv - ev 
        all_mispricings[c] = misprice
        if abs(misprice) > misprice_threshold:
            mispriced_vols[c] = misprice
    most_mispriced = [c[0] for c in sorted(mispriced_vols.items(), key=lambda c: abs(c[1]), reverse=True)]

    short_qs = {k: 0 for k in calls + [underlying]}
    long_qs = {k: 0 for k in calls + [underlying]}

    for c in most_mispriced:
        if mispriced_vols[c] > 0:
            short_qs[c] = min(20, max_call_pos + posns[c])
        else:
            long_qs[c] = max(-20, posns[c] - max_call_pos)
    
    orders = {k: [] for k in calls + [underlying]}

    total_delta = 0
    total_gamma = 0
    deltas, gammas, call_poss = {underlying: 1}, {underlying: 0}, {}
    for call in calls:
        strike = int(call.split("_")[-1])
        delta = bsmodel.delta(strike, vol)
        gamma = bsmodel.gamma(strike, vol)
        call_pos = -short_qs[call] - long_qs[call] + posns[call]
        total_delta += delta * call_pos
        total_gamma += gamma * call_pos
        deltas[call] = delta
        gammas[call] = gamma
        call_poss[call] = call_pos
    total_delta += posns[underlying]

    hedge_limits = {underlying: min(20, max_stock_pos - posns[underlying] if total_delta < 0 else max_stock_pos + posns[underlying])}
    for c in calls:
        hedge_limits[c] = min(20, max_call_pos - posns[c] if total_delta < 0 else max_call_pos + posns[c])


    if abs(total_delta) > delta_hedge_threshold:
        result = optimize_DG(total_delta, total_gamma, deltas, gammas, all_mispricings, misprice_threshold, hedge_limits)
        if result:
            for i in range(2):
                if result["coeffs"][i] < 0:
                    short_qs[result["keys"][i]] -= result["coeffs"][i]
                else:
                    long_qs[result["keys"][i]] -= result["coeffs"][i]

    for k in calls + [underlying]:

        orders[k] = ORDER_ENGINE(
            k,
            short_qs[k],
            long_qs[k],
            outstanding_bids[k],
            outstanding_asks[k],
            round(mids[k]) if posns[k] < 0 else math.floor(mids[k]),
            round(mids[k]) if posns[k] < 0 else math.floor(mids[k]),
            state.order_depths[k],
        )

    return orders


def get_line_pandas(file_path, line_number):
    """
    Retrieves a specific line from a CSV file using pandas.

    Args:
        file_path (str): The path to the CSV file.
        line_number (int): The line number to retrieve (1-based index).

    Returns:
        pd.Series: A pandas Series representing the row, or None if the line number is invalid.
    """
    try:
        df = pd.read_csv(file_path, skiprows=line_number - 1, nrows=1, header=None)
        return df.iloc[0]
    except IndexError:
        return None


def macarons(state, mac_traderData):
    
    # convs = get_line_pandas("data/round4/observations_round_4_day_3.csv", round(state.timestamp / 100 + 2))
    # state.observations.conversionObservations["MAGNIFICENT_MACARONS"] = ConversionObservation(convs[1], convs[2], convs[3], convs[4], convs[5], convs[6], convs[7])
    obs = state.observations.conversionObservations.get("MAGNIFICENT_MACARONS", None)
    if obs is None:
        return [], 0, ""
    sun_index = obs.sunlightIndex
    if not mac_traderData:
        return [], 0, str(sun_index)
    
    max_lag = 10
    sun_values = [float(p) for p in mac_traderData.split(",")[1 - max_lag:]] + [sun_index]
    slope = (sun_values[-1] - sun_values[0]) / len(sun_values)


    MEAN_REVERSION_THRESHOLD = 650
    CSI_THRESHOLD = 49.75
    # SLOPE_THRESHOLD = 0.075
    SLOPE_THRESHOLD = 0.0

    mm_vol = state.position.get("MAGNIFICENT_MACARONS", 0)
    convs = 0

    outstanding_asks = sorted(state.order_depths["MAGNIFICENT_MACARONS"].sell_orders.keys())
    outstanding_bids = sorted(state.order_depths["MAGNIFICENT_MACARONS"].buy_orders.keys())
    mid = (outstanding_bids[-1] + outstanding_asks[0]) / 2
    
    order = []

    buy_price = obs.askPrice + obs.transportFees + obs.importTariff

    ### DECISION TREE ###
    if sun_index < CSI_THRESHOLD:
        if slope > SLOPE_THRESHOLD:
            order.append(Order("MAGNIFICENT_MACARONS", round(mid), -(75 + mm_vol)))
        else:
            if mm_vol < 0:
                convs = min(10, -mm_vol)
                order.append(Order("MAGNIFICENT_MACARONS", math.floor(buy_price) - 1, max(0, 65 - mm_vol)))
            else:
                order.append(Order("MAGNIFICENT_MACARONS", math.floor(buy_price) - 1, 75 - mm_vol))
    else:
        if mid > MEAN_REVERSION_THRESHOLD:
            order.append(Order("MAGNIFICENT_MACARONS", round(mid), -(75 + mm_vol)))
        else:
            if mm_vol < 0:
                convs = min(10, -mm_vol)
                order.append(Order("MAGNIFICENT_MACARONS", math.floor(buy_price) - 1, max(0, 65 - mm_vol)))
            else:
                order.append(Order("MAGNIFICENT_MACARONS", math.floor(buy_price) - 1, 75 - mm_vol))
                

    return order, convs, ",".join(map(str, sun_values))


def croissants(state, croissants_traderData):

    str_return = croissants_traderData
    if not croissants_traderData:
        str_return = "holdoff"
    if "CROISSANTS" in state.market_trades:
        for t in state.market_trades["CROISSANTS"]:
            if t.buyer == "Olivia" and t.seller == "Caesar":
                str_return = "golong"
            elif t.buyer == "Caesar" and t.seller == "Olivia":
                str_return = "goshort"
            print("!!", t.buyer, "!!")

    outstanding_asks = sorted(state.order_depths["CROISSANTS"].sell_orders.keys())
    outstanding_bids = sorted(state.order_depths["CROISSANTS"].buy_orders.keys())
    mid = round((outstanding_bids[-1] + outstanding_asks[0]) / 2)
    c_vol = state.position.get("CROISSANTS", 0)

    order = []
    if str_return == "golong":
        order.append(Order("CROISSANTS", mid + 2, 250 - c_vol))
    elif str_return == "goshort":
        order.append(Order("CROISSANTS", mid - 2, -(c_vol + 250)))


    return order, str_return


class Trader:

    def run(self, state: TradingState):

        orders = {}

        all_traderData = state.traderData.split(";") if state.traderData else ["", "", "", "", "", ""]

        orders["RAINFOREST_RESIN"] = RainforestResinMM(state).make_orders()
        orders["KELP"], kelp_traderData = KelpMM(state, all_traderData[0]).make_orders()
        orders["SQUID_INK"], squid_ink_traderData = SQUID_INK_MM(state, all_traderData[1])

        com_ords_r2, pb1_traderData, pb2_traderData = arb_orders_for_round_2(state, all_traderData[2], all_traderData[3])
        for k in com_ords_r2:
            orders[k] = com_ords_r2[k]

        com_ords_r3 = volcano_orders(state)
        for k in com_ords_r3:
            orders[k] = com_ords_r3[k]

        orders["MAGNIFICENT_MACARONS"], convs, mac_traderData = macarons(state, all_traderData[4])

        orders["CROISSANTS"], croissants_traderData = croissants(state, all_traderData[5])
            
        return orders, convs, ";".join([kelp_traderData, squid_ink_traderData, pb1_traderData, pb2_traderData, mac_traderData, croissants_traderData])