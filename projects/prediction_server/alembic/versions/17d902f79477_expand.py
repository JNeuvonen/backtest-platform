"""Expand

Revision ID: 17d902f79477
Revises: 21940126744f
Create Date: 2024-05-31 13:24:04.552459

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "17d902f79477"
down_revision: Union[str, None] = "21940126744f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column(
        "strategy",
        sa.Column("last_loan_attempt_fail_time_ms", sa.BigInteger(), nullable=True),
    )
    op.add_column(
        "strategy", sa.Column("is_no_loan_available_err", sa.Boolean(), nullable=True)
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column("strategy", "is_no_loan_available_err")
    op.drop_column("strategy", "last_loan_attempt_fail_time_ms")
