from alembic import op
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision = 'f033e2951e9a'
down_revision = '48514981cdd2'
branch_labels = None
depends_on = None


def upgrade():
    # Alter the 'embedding' column to remove the size restriction
    op.alter_column(
        'documents',
        'embedding',
        type_=Vector(),  # Remove size specification
        existing_type=Vector(1560),  # Previous type with size
        nullable=False,
    )


def downgrade():
    # Revert the 'embedding' column to have a size restriction
    op.alter_column(
        'documents',
        'embedding',
        type_=Vector(1560),  # Restore size specification
        existing_type=Vector(),  # No size specification
        nullable=False,
    )
