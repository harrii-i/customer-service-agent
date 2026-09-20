"""Turn `users` into real accounts: name, email, password hash, updated_at.

The columns are added nullable, backfilled, then made NOT NULL. Adding a NOT
NULL column straight onto a populated table fails, and dropping the rows
instead would take their conversations with them through the cascade.

Rows that predate authentication were browser-generated identities, not
accounts. They keep their conversations and are given a placeholder identity
with an unusable password hash, so nobody can sign in as one. They are
recognisable by their `@legacy.invalid` address — `.invalid` is reserved by
RFC 2606 and can never be a real domain.

Revision ID: 0002_user_auth
Revises: 0001_baseline
"""

import sqlalchemy as sa
from alembic import op

revision = "0002_user_auth"
down_revision = "0001_baseline"
branch_labels = None
depends_on = None

# Not a hash of anything. No password can produce it, so verification always
# fails — which is what we want for an account that was never signed up for.
UNUSABLE_PASSWORD_HASH = "!unusable"


def upgrade() -> None:
    op.add_column("users", sa.Column("name", sa.String(length=120), nullable=True))
    op.add_column("users", sa.Column("email", sa.String(length=320), nullable=True))
    op.add_column(
        "users", sa.Column("password_hash", sa.String(length=255), nullable=True)
    )
    op.add_column(
        "users",
        sa.Column(
            "updated_at", sa.DateTime(timezone=True),
            server_default=sa.text("now()"), nullable=True,
        ),
    )

    op.execute(
        sa.text(
            """
            UPDATE users
               SET name = COALESCE(name, 'Legacy user'),
                   email = COALESCE(email, 'legacy+' || id::text || '@legacy.invalid'),
                   password_hash = COALESCE(password_hash, :unusable),
                   updated_at = COALESCE(updated_at, created_at, now())
            """
        ).bindparams(unusable=UNUSABLE_PASSWORD_HASH)
    )

    for column in ("name", "email", "password_hash", "updated_at"):
        op.alter_column("users", column, nullable=False)

    # The index is the uniqueness rule. Checking in Python only would let two
    # concurrent signups for one address both pass and both insert.
    op.create_unique_constraint("uq_users_email", "users", ["email"])


def downgrade() -> None:
    op.drop_constraint("uq_users_email", "users", type_="unique")
    for column in ("updated_at", "password_hash", "email", "name"):
        op.drop_column("users", column)
