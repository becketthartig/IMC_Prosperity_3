from datamodel import OrderDepth, UserId, TradingState, Order
import math

class Trader:

    def run(self, state: TradingState):

        orders = {"RAINFOREST_RESIN": []}

        rr_vol = state.position.get("RAINFOREST_RESIN", 0)

        if rr_vol != 0:
            orders["RAINFOREST_RESIN"].append(Order("RAINFOREST_RESIN", 10000, -rr_vol))

        
        buy_q = max(0, rr_vol)
        sell_q = min(0, rr_vol)
        for _ in range(3):
            added_q = math.floor(math.log(50 - buy_q) * 2.6)
            subbed_q = -math.floor(math.log(50 - abs(sell_q)) * 2.6)
            orders["RAINFOREST_RESIN"].append(Order("RAINFOREST_RESIN", 
                                                    10000 - math.floor(buy_q ** 2 / 400) - 1, 
                                                    added_q))
            orders["RAINFOREST_RESIN"].append(Order("RAINFOREST_RESIN", 
                                                    10000 + math.floor(sell_q ** 2 / 400) + 1, 
                                                    subbed_q))
            buy_q += added_q
            sell_q += subbed_q

        print(orders)


        return orders, 0, "Hello"