import csv
import datetime
from decimal import Decimal

import ExchangeData


PRICE_ROW = 1
VOLUME_ROW = 5
DATE_ROW = 0
ENCODING = 'utf-8'


def exchange_data_list_from_csv(csv_path: str, exchange_data_type: ExchangeData.ExchangeDataType, name: str) -> list:
    my_list = []

    for row in get_next_csv_row(csv_path):
        my_list.append(exchange_data_from_csv_row(row, exchange_data_type, name))

    return my_list


def get_next_csv_row(csv_path: str) -> list:
    with open(csv_path, "r", encoding=ENCODING) as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            yield row


def exchange_data_from_csv_row(csv_row: list, exchange_data_type: ExchangeData.ExchangeDataType, name: str) -> ExchangeData.ExchangeData:
    price = parse_number(csv_row[PRICE_ROW])
    volume = parse_number(csv_row[VOLUME_ROW])
    date = datetime.datetime.strptime(csv_row[DATE_ROW], '%d.%m.%Y')

    return ExchangeData.ExchangeData(name, exchange_data_type, price, volume, date)


def parse_number(volume_string: str) -> Decimal:
    if len(volume_string) == 0:
        return Decimal(0)

    volume_string = volume_string.replace('.', '').replace(',', '.').strip()
    match volume_string[-1].lower():
        case 'm':
            return Decimal(volume_string[:-1]) * 1_000_000

        case 'k':
            return Decimal(volume_string[:-1]) * 1_000

        case _:
            return Decimal(volume_string)


def set_row_numbers(price_row: int = 1, volume_row: int = 5, date_row: int = 0) -> None:
    global PRICE_ROW, VOLUME_ROW, DATE_ROW
    PRICE_ROW = price_row
    VOLUME_ROW = volume_row
    DATE_ROW = date_row


def set_encoding(encoding: str):
    global ENCODING
    ENCODING = encoding
