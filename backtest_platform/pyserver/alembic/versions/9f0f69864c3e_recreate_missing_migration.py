"""Recreate missing migration

Revision ID: 9f0f69864c3e
Revises: 5dbb7842f528
Create Date: 2024-04-14 08:14:34.213304

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9f0f69864c3e'
down_revision: Union[str, None] = '5dbb7842f528'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
