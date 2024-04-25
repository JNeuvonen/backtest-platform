"""Add two fields to backtest table

Revision ID: df98eb59d832
Revises: 03625184ad1c
Create Date: 2024-04-25 16:11:07.965133

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'df98eb59d832'
down_revision: Union[str, None] = '03625184ad1c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('backtest', sa.Column('is_long_short_strategy', sa.Boolean(), nullable=True))
    op.add_column('backtest', sa.Column('is_ml_based_strategy', sa.Boolean(), nullable=True))
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('backtest', 'is_ml_based_strategy')
    op.drop_column('backtest', 'is_long_short_strategy')
    # ### end Alembic commands ###
