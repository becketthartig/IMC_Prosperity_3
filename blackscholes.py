from statistics import NormalDist
import math

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
    
    def implied_vol(self, K, C, precision_limit=1e-6, iter_limit=100):
        low = 1e-6
        high = 200
        mid = 0
        for _ in range(iter_limit):
            mid = (low + high) / 2
            price = self.call_price(K, mid)
            if abs(price - C) < precision_limit:
                return mid
            elif price < C:
                low = mid
            else:
                high = mid
        return mid 
        
    
    
bsmodel = BLACK_SCHOLES_CALC(10000, 5)
print(bsmodel.implied_vol(10250, 50))


