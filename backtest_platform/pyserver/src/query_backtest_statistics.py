from typing import Dict
from sqlalchemy import BigInteger, Column, Float, ForeignKey, Integer
from log import LogExceptionContext
from orm import Base, Session


class BacktestStatistics(Base):
    __tablename__ = "backtest_statistics"

    id = Column(Integer, primary_key=True)
    backtest_id = Column(Integer, ForeignKey("backtest.id"))

    profit_factor = Column(Float)
    long_side_profit_factor = Column(Float)
    short_side_profit_factor = Column(Float)
    gross_profit = Column(Float)
    gross_loss = Column(Float)
    start_balance = Column(Float)
    end_balance = Column(Float)
    result_perc = Column(Float)
    take_profit_threshold_perc = Column(Float)
    stop_loss_threshold_perc = Column(Float)
    best_trade_result_perc = Column(Float)
    worst_trade_result_perc = Column(Float)
    best_long_side_trade_result_perc = Column(Float)
    worst_long_side_trade_result_perc = Column(Float)
    best_short_side_trade_result_perc = Column(Float)
    worst_short_side_trade_result_perc = Column(Float)
    buy_and_hold_result_net = Column(Float)
    buy_and_hold_result_perc = Column(Float)
    sharpe_ratio = Column(Float)
    probabilistic_sharpe_ratio = Column(Float)
    share_of_winning_trades_perc = Column(Float)
    share_of_losing_trades_perc = Column(Float)
    max_drawdown_perc = Column(Float)
    benchmark_drawdown_perc = Column(Float)
    cagr = Column(Float)
    market_exposure_time = Column(Float)
    risk_adjusted_return = Column(Float)
    buy_and_hold_cagr = Column(Float)
    slippage_perc = Column(Float)
    short_fee_hourly = Column(Float)
    trading_fees_perc = Column(Float)

    trade_count = Column(Integer)
    mean_hold_time_sec = Column(Integer)
    mean_return_perc = Column(Float)


class BacktestStatisticsQuery:
    @staticmethod
    def create_entry(fields: Dict):
        with LogExceptionContext():
            with Session() as session:
                entry = BacktestStatistics(**fields)
                session.add(entry)
                session.commit()
                return entry.id

    @staticmethod
    def fetch_backtest_by_id(backtest_id: int):
        with LogExceptionContext():
            with Session() as session:
                backtest_data = (
                    session.query(BacktestStatistics)
                    .filter(BacktestStatistics.backtest_id == backtest_id)
                    .first()
                )
                return backtest_data
