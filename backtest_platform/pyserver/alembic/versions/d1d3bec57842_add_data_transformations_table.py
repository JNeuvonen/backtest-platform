"""Add data_transformations table

Revision ID: d1d3bec57842
Revises: 9f0f69864c3e
Create Date: 2024-04-14 08:22:06.920137

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "d1d3bec57842"
down_revision: Union[str, None] = "9f0f69864c3e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
