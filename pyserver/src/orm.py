from sqlalchemy import (
    create_engine,
)
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import create_engine
from config import append_app_data_path
from constants import DATASET_UTILS_DB_PATH

DATABASE_URI = f"sqlite:///{append_app_data_path(DATASET_UTILS_DB_PATH)}"

engine = create_engine(DATABASE_URI)
Base = declarative_base()
Session = sessionmaker(bind=engine)


def db_delete_all_data():
    with Session() as session:
        for table in reversed(Base.metadata.sorted_tables):
            session.execute(table.delete())
        session.commit()


def drop_tables():
    Base.metadata.drop_all(engine)


def create_tables():
    Base.metadata.create_all(engine)
