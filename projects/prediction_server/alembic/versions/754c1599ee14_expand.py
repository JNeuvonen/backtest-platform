"""Expand

Revision ID: 754c1599ee14
Revises: 86740acc428e
Create Date: 2024-06-22 12:41:02.439639

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '754c1599ee14'
down_revision: Union[str, None] = '86740acc428e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_foreign_key(None, 'long_short_pair', 'long_short_ticker', ['buy_ticker_id'], ['id'])
    op.create_foreign_key(None, 'long_short_pair', 'long_short_group', ['long_short_group_id'], ['id'])
    op.create_foreign_key(None, 'long_short_pair', 'long_short_ticker', ['sell_ticker_id'], ['id'])
    op.create_foreign_key(None, 'long_short_pair', 'trade', ['sell_side_trade_id'], ['id'])
    op.create_foreign_key(None, 'long_short_pair', 'trade', ['buy_side_trade_id'], ['id'])
    op.create_foreign_key(None, 'refetch_strategy_signal', 'strategy', ['strategy_id'], ['id'])
    op.add_column('strategy_group', sa.Column('last_adaptive_group_recalc', sa.DateTime(), nullable=True))
    op.add_column('strategy_group', sa.Column('is_auto_adaptive_group', sa.Boolean(), nullable=True))
    op.add_column('strategy_group', sa.Column('num_symbols_for_auto_adaptive', sa.Integer(), nullable=True))
    op.add_column('strategy_group', sa.Column('num_days_for_group_recalc', sa.Integer(), nullable=True))
    op.add_column('strategy_group', sa.Column('enter_trade_code', sa.String(), nullable=True))
    op.add_column('strategy_group', sa.Column('exit_trade_code', sa.String(), nullable=True))
    op.add_column('strategy_group', sa.Column('fetch_datasources_code', sa.String(), nullable=True))
    op.add_column('strategy_group', sa.Column('candle_interval', sa.String(), nullable=True))
    op.add_column('strategy_group', sa.Column('priority', sa.Integer(), nullable=True))
    op.add_column('strategy_group', sa.Column('num_req_klines', sa.Integer(), nullable=True))
    op.add_column('strategy_group', sa.Column('kline_size_ms', sa.Integer(), nullable=True))
    op.add_column('strategy_group', sa.Column('minimum_time_between_trades_ms', sa.Integer(), nullable=True))
    op.add_column('strategy_group', sa.Column('maximum_klines_hold_time', sa.Integer(), nullable=True))
    op.add_column('strategy_group', sa.Column('allocated_size_perc', sa.Float(), nullable=True))
    op.add_column('strategy_group', sa.Column('take_profit_threshold_perc', sa.Float(), nullable=True))
    op.add_column('strategy_group', sa.Column('stop_loss_threshold_perc', sa.Float(), nullable=True))
    op.add_column('strategy_group', sa.Column('use_time_based_close', sa.Boolean(), nullable=True))
    op.add_column('strategy_group', sa.Column('use_profit_based_close', sa.Boolean(), nullable=True))
    op.add_column('strategy_group', sa.Column('use_stop_loss_based_close', sa.Boolean(), nullable=True))
    op.add_column('strategy_group', sa.Column('use_taker_order', sa.Boolean(), nullable=True))
    op.add_column('strategy_group', sa.Column('should_calc_stops_on_pred_serv', sa.Boolean(), nullable=True))
    op.add_column('strategy_group', sa.Column('is_leverage_allowed', sa.Boolean(), nullable=True))
    op.add_column('strategy_group', sa.Column('is_short_selling_strategy', sa.Boolean(), nullable=True))
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('strategy_group', 'is_short_selling_strategy')
    op.drop_column('strategy_group', 'is_leverage_allowed')
    op.drop_column('strategy_group', 'should_calc_stops_on_pred_serv')
    op.drop_column('strategy_group', 'use_taker_order')
    op.drop_column('strategy_group', 'use_stop_loss_based_close')
    op.drop_column('strategy_group', 'use_profit_based_close')
    op.drop_column('strategy_group', 'use_time_based_close')
    op.drop_column('strategy_group', 'stop_loss_threshold_perc')
    op.drop_column('strategy_group', 'take_profit_threshold_perc')
    op.drop_column('strategy_group', 'allocated_size_perc')
    op.drop_column('strategy_group', 'maximum_klines_hold_time')
    op.drop_column('strategy_group', 'minimum_time_between_trades_ms')
    op.drop_column('strategy_group', 'kline_size_ms')
    op.drop_column('strategy_group', 'num_req_klines')
    op.drop_column('strategy_group', 'priority')
    op.drop_column('strategy_group', 'candle_interval')
    op.drop_column('strategy_group', 'fetch_datasources_code')
    op.drop_column('strategy_group', 'exit_trade_code')
    op.drop_column('strategy_group', 'enter_trade_code')
    op.drop_column('strategy_group', 'num_days_for_group_recalc')
    op.drop_column('strategy_group', 'num_symbols_for_auto_adaptive')
    op.drop_column('strategy_group', 'is_auto_adaptive_group')
    op.drop_column('strategy_group', 'last_adaptive_group_recalc')
    op.drop_constraint(None, 'refetch_strategy_signal', type_='foreignkey')
    op.drop_constraint(None, 'long_short_pair', type_='foreignkey')
    op.drop_constraint(None, 'long_short_pair', type_='foreignkey')
    op.drop_constraint(None, 'long_short_pair', type_='foreignkey')
    op.drop_constraint(None, 'long_short_pair', type_='foreignkey')
    op.drop_constraint(None, 'long_short_pair', type_='foreignkey')
    # ### end Alembic commands ###