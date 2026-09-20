"""User lookups and creation.

`get_or_create_user` is deliberately gone. It existed so a stale browser id
could materialise an account; with real authentication, an account is created
by signing up and by nothing else. Leaving it would mean any UUID presented to
the API still conjures a user.
"""

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.models import User


def normalise_email(email: str) -> str:
    """Trimmed and lower-cased. Applied on every write *and* every lookup, so
    the stored value and the unique index agree — "Rahul@x.com" and
    "rahul@x.com" are one account, not two."""
    return email.strip().lower()


def create_user(db: Session, name: str, email: str, password_hash: str) -> User:
    user = User(
        name=name.strip(),
        email=normalise_email(email),
        password_hash=password_hash,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_user(db: Session, user_id: uuid.UUID) -> User | None:
    return db.get(User, user_id)


def get_by_email(db: Session, email: str) -> User | None:
    stmt = select(User).where(User.email == normalise_email(email))
    return db.scalars(stmt).first()
