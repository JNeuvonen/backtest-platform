from sqlalchemy import Column, ForeignKey, Integer, String
from orm import Base


class LongShortInfo(Base):
    __tablename__ = "long_short_info"

    id = Column(Integer, primary_key=True)
    backtest_id = Column(Integer, ForeignKey("backtest.id"))

    datasets = Column(String)
    data_transformations = Column(String)
