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
    
    def implied_vol(self, K, C, precision_limit=1e-7, iter_limit=200):
        low = 1e-7
        high = 2
        mid = 0
        for i in range(iter_limit):
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

        
    

bsmodel = BLACK_SCHOLES_CALC(202.52, 3, 0.0399)
# print(bsmodel.implied_vol(202.5, 4.398447041271453))
print(bsmodel.gamma(202.5, 0.5884))
