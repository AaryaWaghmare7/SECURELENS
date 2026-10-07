"""Create the SecureLens account and analysis workspace schema."""
from alembic import op
import sqlalchemy as sa

revision = "0001_workspace"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table("users", sa.Column("id", sa.String(36), primary_key=True), sa.Column("name", sa.String(100), nullable=False),
                    sa.Column("email", sa.String(254), nullable=False), sa.Column("password_hash", sa.Text(), nullable=False),
                    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False), sa.Column("auth_version", sa.Integer(), nullable=False))
    op.create_index("ix_users_email", "users", ["email"], unique=True)
    op.create_table("analyses", sa.Column("id", sa.String(36), primary_key=True),
                    sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
                    sa.Column("analysis_type", sa.String(20), nullable=False), sa.Column("filename", sa.String(200), nullable=False),
                    sa.Column("status", sa.String(20), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
                    sa.Column("result", sa.JSON(), nullable=False), sa.Column("report_key", sa.String(200), nullable=False),
                    sa.Column("image_keys", sa.JSON(), nullable=False), sa.Column("images_expire_at", sa.DateTime(timezone=True)))
    op.create_index("ix_analyses_user_id", "analyses", ["user_id"])
    op.create_index("ix_analyses_created_at", "analyses", ["created_at"])
    op.create_table("user_preferences", sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
                    sa.Column("retain_images", sa.Boolean(), nullable=False))
    op.create_table("cookie_consents", sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
                    sa.Column("essential", sa.Boolean(), nullable=False), sa.Column("authentication", sa.Boolean(), nullable=False),
                    sa.Column("analytics", sa.Boolean(), nullable=False), sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False))


def downgrade():
    for table in ("cookie_consents", "user_preferences", "analyses", "users"):
        op.drop_table(table)
