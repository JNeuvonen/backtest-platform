def get_backtest_profit_factor_comp(trades):
    gross_profit = 0.0
    gross_loss = 0.0

    for item in trades:
        if item["net_result"] >= 0.0:
            gross_profit += item["net_result"]
        else:
            gross_loss += item["net_result"]

    return gross_profit - gross_loss, gross_profit, gross_loss
