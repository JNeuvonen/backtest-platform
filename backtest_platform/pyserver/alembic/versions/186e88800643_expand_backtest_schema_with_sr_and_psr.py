"""Expand backtest schema with SR and PSR

Revision ID: 186e88800643
Revises: 
Create Date: 2024-04-16 15:26:39.419740

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '186e88800643'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('backtest', sa.Column('sharpe_ratio', sa.Float(), nullable=True))
    op.add_column('backtest', sa.Column('probabilistic_sharpe_ratio', sa.Float(), nullable=True))
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('backtest', 'probabilistic_sharpe_ratio')
    op.drop_column('backtest', 'sharpe_ratio')
    # ### end Alembic commands ###
