from sqlalchemy import Column, Integer, String, DateTime, Float, Text
from sqlalchemy.ext.declarative import declarative_base
from alembic import op
import sqlalchemy as sa

def upgrade():
    op.create_table(
        'features',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('formula', sa.Text(), nullable=False),
        sa.Column('type', sa.String(50), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('mean', sa.Float(), nullable=True),
        sa.Column('std', sa.Float(), nullable=True),
        sa.Column('price_correlation', sa.Float(), nullable=True),
        sa.Column('returns_correlation', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes
    op.create_index('ix_features_symbol', 'features', ['symbol'])
    op.create_index('ix_features_type', 'features', ['type'])

def downgrade():
    op.drop_index('ix_features_type')
    op.drop_index('ix_features_symbol')
    op.drop_table('features')