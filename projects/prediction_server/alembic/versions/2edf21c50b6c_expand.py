"""Expand

Revision ID: 2edf21c50b6c
Revises: 5f4c83048ac3
Create Date: 2024-05-26 10:32:48.008981

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "2edf21c50b6c"
down_revision: Union[str, None] = "5f4c83048ac3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint("strategy_strategy_group_key", "strategy", type_="unique")
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_unique_constraint(
        "strategy_strategy_group_key", "strategy", ["strategy_group"]
    )
