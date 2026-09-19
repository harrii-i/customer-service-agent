import uuid

from sqlalchemy.orm import Session

from app.database.models import User


def create_user(db: Session) -> User:
    user = User()
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_user(db: Session, user_id: uuid.UUID) -> User | None:
    return db.get(User, user_id)


def get_or_create_user(db: Session, user_id: uuid.UUID) -> User:
    """The frontend keeps its user id in localStorage, so a returning browser
    can present an id the database no longer has (e.g. a reset dev DB). Rather
    than 404 the whole app, materialise that id."""
    user = db.get(User, user_id)
    if user is None:
        user = User(id=user_id)
        db.add(user)
        db.commit()
        db.refresh(user)
    return user
