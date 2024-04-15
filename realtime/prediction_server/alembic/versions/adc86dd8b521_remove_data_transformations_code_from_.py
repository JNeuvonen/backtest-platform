"""Remove data_transformations_code from strategy

Revision ID: adc86dd8b521
Revises: 
Create Date: 2024-04-15 15:55:32.617274

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'adc86dd8b521'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column('slackbot', 'created_at',
               existing_type=postgresql.TIMESTAMP(),
               nullable=True,
               existing_server_default=sa.text('now()'))
    op.alter_column('slackbot', 'updated_at',
               existing_type=postgresql.TIMESTAMP(),
               nullable=True,
               existing_server_default=sa.text('now()'))
    op.alter_column('slackbot', 'name',
               existing_type=sa.VARCHAR(),
               nullable=True)
    op.drop_column('strategy', 'data_transformations_code')
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('strategy', sa.Column('data_transformations_code', sa.VARCHAR(), autoincrement=False, nullable=False))
    op.alter_column('slackbot', 'name',
               existing_type=sa.VARCHAR(),
               nullable=False)
    op.alter_column('slackbot', 'updated_at',
               existing_type=postgresql.TIMESTAMP(),
               nullable=False,
               existing_server_default=sa.text('now()'))
    op.alter_column('slackbot', 'created_at',
               existing_type=postgresql.TIMESTAMP(),
               nullable=False,
               existing_server_default=sa.text('now()'))
    # ### end Alembic commands ###
