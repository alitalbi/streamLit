import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class pnl:
    def __init__(self):
        self.pnl = None
        self.volatility = None

    def compute_pnl(self,trade_log):
        trade_log["pq_acc"], trade_log["q_acc"] = abs((trade_log.price * trade_log.q)).cumsum(), abs(
            trade_log["q"]).cumsum()
        trade_log["avg_price"] = trade_log["pq_acc"] / trade_log["q_acc"]
        trade_log["pnl"] = np.zeros(len(trade_log))
        for index in range(len(trade_log)):
            if index == 0:
                continue
            trade_log["pnl"][index] = (trade_log["avg_price"][index - 1] - trade_log["price"][index]) * trade_log.q[
                index]
        return trade_log

    def risk_metrics(self,trade_log):
        trades = [trade for trade in range(1, len(trade_log) + 1)]
        trade_log["price_std"] = ((trade_log["price"] - trade_log["avg_price"]) ** 2).cumsum() / trades
        trade_log["pnl_std"] = trade_log["pnl"].cumsum() / trades
        self.volatility = trade_log.pnl_std


if __name__ == "__main__":
    compute_pnl = pnl()
    # ------------------------------------- trade logs ------------------------------------------------------
    trades_p = [10,15,5,25,3,41,29,23]
    trades_q = [4,1,-4,4,-3,13,43,-48]
    trade_log = pd.DataFrame({"q":trades_q,"price":trades_p})
    trade_log["way"] = np.where(trade_log["q"]>0,1,-1)
    #----------------------------------- Trade logs end -----------------------------------------------------

    df = compute_pnl.pnl(trade_log)
    df = compute_pnl.risk_metrics(df)
    print(compute_pnl.volatility)

    print(":)")