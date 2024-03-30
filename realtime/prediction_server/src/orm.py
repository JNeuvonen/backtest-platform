from sqlalchemy import (
    create_engine,
    inspect,
    text,
)
from sqlalchemy.orm import sessionmaker, declarative_base
from config import get_db_uri
from log import LogExceptionContext


engine = create_engine(get_db_uri())
Base = declarative_base()
Session = sessionmaker(bind=engine)


def db_delete_all_data():
    with Session() as session:
        for table in reversed(Base.metadata.sorted_tables):
            session.execute(table.delete())
        session.commit()


def drop_tables(engine):
    with engine.connect() as connection:
        inspector = inspect(engine)
        tables = inspector.get_table_names()

        for table in tables:
            connection.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE;"))


def create_tables():
    with LogExceptionContext(success_log_msg="Created tables"):
        Base.metadata.create_all(engine)


def test_db_conn():
    with LogExceptionContext(success_log_msg="DB connection established."):
        try:
            with Session() as session:
                _ = session.execute(text("SELECT * FROM information_schema.tables;"))
            print("Connection to the DB successful.")
        except Exception as e:
            raise Exception(f"Unable to connect to the DB: {str(e)}")
