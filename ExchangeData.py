import datetime
from decimal import Decimal
from enum import Enum


class ExchangeDataType(Enum):
    Stock = 0
    Index = 1


class ExchangeData:
    def __init__(self, name: str, exchange_data_type: ExchangeDataType, price: Decimal, volume: Decimal, date: datetime.date):
        self.name = name
        self.exchange_data_type = exchange_data_type
        self.price = price
        self.volume = volume
        self.date = date

    def __str__(self):
        return f'{self.name} ({self.exchange_data_type}) {self.price}, volume {self.volume}, on {self.date}'

    def __repr__(self):
        return self.__str__()
