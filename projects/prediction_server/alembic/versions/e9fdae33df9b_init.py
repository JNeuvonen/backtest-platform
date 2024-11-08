"""init

Revision ID: e9fdae33df9b
Revises: 
Create Date: 2024-05-05 12:57:00.862380

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "e9fdae33df9b"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    if not op.get_bind().dialect.has_table(op.get_bind(), "account"):
        op.create_table(
            "account",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.Column("updated_at", sa.DateTime(), nullable=True),
            sa.Column("name", sa.String(), nullable=True),
            sa.Column("max_ratio_of_longs_to_nav", sa.Float(), nullable=True),
            sa.Column("max_debt_ratio", sa.Float(), nullable=True),
            sa.Column("prevent_all_trading", sa.Boolean(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("name"),
        )

    if not op.get_bind().dialect.has_table(op.get_bind(), "api_keys"):
        op.create_table(
            "api_keys",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.Column("updated_at", sa.DateTime(), nullable=True),
            sa.Column("key", sa.String(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("key"),
        )

    if not op.get_bind().dialect.has_table(op.get_bind(), "cloud_log"):
        op.create_table(
            "cloud_log",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.Column("message", sa.String(), nullable=True),
            sa.Column("level", sa.String(), nullable=True),
            sa.Column("source_program", sa.Integer(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )
    if not op.get_bind().dialect.has_table(op.get_bind(), "slackbot"):
        op.create_table(
            "slackbot",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.Column("updated_at", sa.DateTime(), nullable=True),
            sa.Column("name", sa.String(), nullable=True),
            sa.Column("webhook_uri", sa.String(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("name"),
        )

    if not op.get_bind().dialect.has_table(op.get_bind(), "strategy"):
        op.create_table(
            "strategy",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("active_trade_id", sa.Integer(), nullable=True),
            sa.Column("name", sa.String(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.Column("updated_at", sa.DateTime(), nullable=True),
            sa.Column("symbol", sa.String(), nullable=False),
            sa.Column("base_asset", sa.String(), nullable=False),
            sa.Column("quote_asset", sa.String(), nullable=False),
            sa.Column("enter_trade_code", sa.String(), nullable=False),
            sa.Column("exit_trade_code", sa.String(), nullable=False),
            sa.Column("fetch_datasources_code", sa.String(), nullable=False),
            sa.Column("trade_quantity_precision", sa.Integer(), nullable=False),
            sa.Column("priority", sa.Integer(), nullable=False),
            sa.Column("kline_size_ms", sa.Integer(), nullable=True),
            sa.Column("prev_kline_ms", sa.Integer(), nullable=True),
            sa.Column("minimum_time_between_trades_ms", sa.Integer(), nullable=True),
            sa.Column("maximum_klines_hold_time", sa.Integer(), nullable=True),
            sa.Column("time_on_trade_open_ms", sa.BigInteger(), nullable=True),
            sa.Column("price_on_trade_open", sa.Float(), nullable=True),
            sa.Column("quantity_on_trade_open", sa.Float(), nullable=True),
            sa.Column("remaining_position_on_trade", sa.Float(), nullable=True),
            sa.Column("allocated_size_perc", sa.Float(), nullable=True),
            sa.Column("take_profit_threshold_perc", sa.Float(), nullable=True),
            sa.Column("stop_loss_threshold_perc", sa.Float(), nullable=True),
            sa.Column("use_time_based_close", sa.Boolean(), nullable=False),
            sa.Column("use_profit_based_close", sa.Boolean(), nullable=False),
            sa.Column("use_stop_loss_based_close", sa.Boolean(), nullable=False),
            sa.Column("use_taker_order", sa.Boolean(), nullable=True),
            sa.Column("should_enter_trade", sa.Boolean(), nullable=True),
            sa.Column("should_close_trade", sa.Boolean(), nullable=True),
            sa.Column("is_on_pred_serv_err", sa.Boolean(), nullable=True),
            sa.Column("is_paper_trade_mode", sa.Boolean(), nullable=True),
            sa.Column("is_leverage_allowed", sa.Boolean(), nullable=True),
            sa.Column("is_short_selling_strategy", sa.Boolean(), nullable=False),
            sa.Column("is_disabled", sa.Boolean(), nullable=True),
            sa.Column("is_in_position", sa.Boolean(), nullable=True),
            sa.ForeignKeyConstraint(
                ["active_trade_id"],
                ["trade.id"],
            ),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("name"),
        )

    if not op.get_bind().dialect.has_table(op.get_bind(), "trade"):
        op.create_table(
            "trade",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("strategy_id", sa.Integer(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.Column("updated_at", sa.DateTime(), nullable=True),
            sa.Column("open_time_ms", sa.BigInteger(), nullable=False),
            sa.Column("close_time_ms", sa.BigInteger(), nullable=True),
            sa.Column("open_price", sa.Float(), nullable=False),
            sa.Column("quantity", sa.Float(), nullable=False),
            sa.Column("close_price", sa.Float(), nullable=True),
            sa.Column("net_result", sa.Float(), nullable=True),
            sa.Column("percent_result", sa.Float(), nullable=True),
            sa.Column("direction", sa.String(), nullable=False),
            sa.Column(
                "profit_history", postgresql.JSON(astext_type=sa.Text()), nullable=False
            ),
            sa.ForeignKeyConstraint(
                ["strategy_id"],
                ["strategy.id"],
            ),
            sa.PrimaryKeyConstraint("id"),
        )

    if not op.get_bind().dialect.has_table(op.get_bind(), "whitelisted_ip"):
        op.create_table(
            "whitelisted_ip",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.Column("updated_at", sa.DateTime(), nullable=True),
            sa.Column("ip", sa.String(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("ip"),
        )

    if not op.get_bind().dialect.has_table(op.get_bind(), "data_transformation"):
        op.create_table(
            "data_transformation",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("strategy_id", sa.Integer(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.Column("updated_at", sa.DateTime(), nullable=True),
            sa.Column("transformation_code", sa.String(), nullable=False),
            sa.ForeignKeyConstraint(
                ["strategy_id"],
                ["strategy.id"],
            ),
            sa.PrimaryKeyConstraint("id"),
        )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table("data_transformation")
    op.drop_table("whitelisted_ip")
    op.drop_table("trade")
    op.drop_table("strategy")
    op.drop_table("slackbot")
    op.drop_table("cloud_log")
    op.drop_table("api_keys")
    op.drop_table("account")
    # ### end Alembic commands ###
