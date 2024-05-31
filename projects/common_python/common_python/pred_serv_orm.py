from contextlib import contextmanager
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    create_engine,
    func,
    text,
)
from sqlalchemy.orm import sessionmaker, declarative_base
from common_python.pred_serv_config import get_db_uri


engine = create_engine(get_db_uri())
Base = declarative_base()
Session = sessionmaker(bind=engine)


class LongShortGroup(Base):
    __tablename__ = "long_short_group"

    id = Column(Integer, primary_key=True)

    name = Column(String, unique=True)
    candle_interval = Column(String)
    buy_cond = Column(String)
    sell_cond = Column(String)
    exit_cond = Column(String)

    num_req_klines = Column(Integer)
    max_simultaneous_positions = Column(Integer)
    klines_until_close = Column(Integer)
    kline_size_ms = Column(Integer)
    loan_retry_wait_time_ms = Column(Integer)

    max_leverage_ratio = Column(Float)
    take_profit_threshold_perc = Column(Float)
    stop_loss_threshold_perc = Column(Float)

    is_disabled = Column(Boolean, default=True)
    is_in_close_only = Column(Boolean, default=False)
    use_time_based_close = Column(Boolean)
    use_profit_based_close = Column(Boolean)
    use_stop_loss_based_close = Column(Boolean)
    use_taker_order = Column(Boolean)


class StrategyGroup(Base):
    __tablename__ = "strategy_group"

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    transformation_ids = Column(String)
    is_disabled = Column(Boolean, default=True)
    is_close_only = Column(Boolean, default=False)


def db_delete_all_data():
    with Session() as session:
        for table in reversed(Base.metadata.sorted_tables):
            session.execute(table.delete())
        session.commit()


def drop_tables(engine):
    metadata = MetaData()
    metadata.reflect(bind=engine)
    metadata.drop_all(bind=engine)


def create_tables():
    LongShortGroup.__table__.create(bind=engine, checkfirst=True)
    StrategyGroup.__table__.create(bind=engine, checkfirst=True)
    tables = Base.metadata.tables
    Base.metadata.create_all(engine)


def test_db_conn():
    try:
        with Session() as session:
            _ = session.execute(text("SELECT * FROM information_schema.tables;"))
        print("Connection to the DB successful.")
    except Exception as e:
        raise Exception(f"Unable to connect to the DB: {str(e)}")


@contextmanager
def TransactionContext():
    session = Session()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        raise
    finally:
        session.close()
