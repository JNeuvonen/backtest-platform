from sqlalchemy import (
    MetaData,
    create_engine,
)
from sqlalchemy.orm import sessionmaker, declarative_base

from config import DATASETS_DB_URI


engine = create_engine(f"sqlite:///{DATASETS_DB_URI}")
Base = declarative_base()
Session = sessionmaker(bind=engine)


def db_delete_all_data():
    with Session() as session:
        for table in reversed(Base.metadata.sorted_tables):
            session.execute(table.delete())
        session.commit()


def drop_tables(engine):
    metadata = MetaData()
    metadata.reflect(bind=engine)
    metadata.drop_all(bind=engine)


def create_sqlite_tables():
    try:
        Base.metadata.create_all(engine)
    except Exception:
        pass
