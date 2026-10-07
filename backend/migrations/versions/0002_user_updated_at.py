"""Add the account update timestamp without replacing existing accounts."""
from alembic import op
import sqlalchemy as sa

revision = "0002_user_updated_at"
down_revision = "0001_workspace"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("users", sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True))
    op.execute(sa.text("UPDATE users SET updated_at = created_at WHERE updated_at IS NULL"))
    with op.batch_alter_table("users") as table:
        table.alter_column("updated_at", existing_type=sa.DateTime(timezone=True), nullable=False)


def downgrade():
    with op.batch_alter_table("users") as table:
        table.drop_column("updated_at")
