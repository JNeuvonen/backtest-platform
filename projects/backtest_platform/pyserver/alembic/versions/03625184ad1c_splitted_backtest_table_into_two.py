"""Splitted backtest table into two

Revision ID: 03625184ad1c
Revises: 5eeb569bb8a8
Create Date: 2024-04-25 11:08:38.102589

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '03625184ad1c'
down_revision: Union[str, None] = '5eeb569bb8a8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('backtest', 'gross_profit')
    op.drop_column('backtest', 'open_long_trade_cond')
    op.drop_column('backtest', 'buy_and_hold_cagr')
    op.drop_column('backtest', 'profit_factor')
    op.drop_column('backtest', 'buy_and_hold_result_net')
    op.drop_column('backtest', 'max_drawdown_perc')
    op.drop_column('backtest', 'share_of_losing_trades_perc')
    op.drop_column('backtest', 'open_short_trade_cond')
    op.drop_column('backtest', 'start_balance')
    op.drop_column('backtest', 'result_perc')
    op.drop_column('backtest', 'share_of_winning_trades_perc')
    op.drop_column('backtest', 'close_short_trade_cond')
    op.drop_column('backtest', 'stop_loss_threshold_perc')
    op.drop_column('backtest', 'market_exposure_time')
    op.drop_column('backtest', 'trading_fees_perc')
    op.drop_column('backtest', 'gross_loss')
    op.drop_column('backtest', 'short_fee_hourly')
    op.drop_column('backtest', 'slippage_perc')
    op.drop_column('backtest', 'worst_trade_result_perc')
    op.drop_column('backtest', 'end_balance')
    op.drop_column('backtest', 'probabilistic_sharpe_ratio')
    op.drop_column('backtest', 'best_trade_result_perc')
    op.drop_column('backtest', 'cagr')
    op.drop_column('backtest', 'close_long_trade_cond')
    op.drop_column('backtest', 'sharpe_ratio')
    op.drop_column('backtest', 'risk_adjusted_return')
    op.drop_column('backtest', 'trade_count')
    op.drop_column('backtest', 'take_profit_threshold_perc')
    op.drop_column('backtest', 'buy_and_hold_result_perc')
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('backtest', sa.Column('buy_and_hold_result_perc', sa.FLOAT(), nullable=True))
    op.add_column('backtest', sa.Column('take_profit_threshold_perc', sa.FLOAT(), nullable=True))
    op.add_column('backtest', sa.Column('trade_count', sa.INTEGER(), nullable=True))
    op.add_column('backtest', sa.Column('risk_adjusted_return', sa.FLOAT(), nullable=True))
    op.add_column('backtest', sa.Column('sharpe_ratio', sa.FLOAT(), nullable=True))
    op.add_column('backtest', sa.Column('close_long_trade_cond', sa.VARCHAR(), nullable=True))
    op.add_column('backtest', sa.Column('cagr', sa.FLOAT(), nullable=True))
    op.add_column('backtest', sa.Column('best_trade_result_perc', sa.FLOAT(), nullable=True))
    op.add_column('backtest', sa.Column('probabilistic_sharpe_ratio', sa.FLOAT(), nullable=True))
    op.add_column('backtest', sa.Column('end_balance', sa.FLOAT(), nullable=True))
    op.add_column('backtest', sa.Column('worst_trade_result_perc', sa.FLOAT(), nullable=True))
    op.add_column('backtest', sa.Column('slippage_perc', sa.FLOAT(), nullable=True))
    op.add_column('backtest', sa.Column('short_fee_hourly', sa.FLOAT(), nullable=True))
    op.add_column('backtest', sa.Column('gross_loss', sa.FLOAT(), nullable=True))
    op.add_column('backtest', sa.Column('trading_fees_perc', sa.FLOAT(), nullable=True))
    op.add_column('backtest', sa.Column('market_exposure_time', sa.FLOAT(), nullable=True))
    op.add_column('backtest', sa.Column('stop_loss_threshold_perc', sa.FLOAT(), nullable=True))
    op.add_column('backtest', sa.Column('close_short_trade_cond', sa.VARCHAR(), nullable=True))
    op.add_column('backtest', sa.Column('share_of_winning_trades_perc', sa.FLOAT(), nullable=True))
    op.add_column('backtest', sa.Column('result_perc', sa.FLOAT(), nullable=True))
    op.add_column('backtest', sa.Column('start_balance', sa.FLOAT(), nullable=True))
    op.add_column('backtest', sa.Column('open_short_trade_cond', sa.VARCHAR(), nullable=True))
    op.add_column('backtest', sa.Column('share_of_losing_trades_perc', sa.FLOAT(), nullable=True))
    op.add_column('backtest', sa.Column('max_drawdown_perc', sa.FLOAT(), nullable=True))
    op.add_column('backtest', sa.Column('buy_and_hold_result_net', sa.FLOAT(), nullable=True))
    op.add_column('backtest', sa.Column('profit_factor', sa.FLOAT(), nullable=True))
    op.add_column('backtest', sa.Column('buy_and_hold_cagr', sa.FLOAT(), nullable=True))
    op.add_column('backtest', sa.Column('open_long_trade_cond', sa.VARCHAR(), nullable=True))
    op.add_column('backtest', sa.Column('gross_profit', sa.FLOAT(), nullable=True))
    # ### end Alembic commands ###