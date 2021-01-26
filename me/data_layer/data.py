# from sqlalchemy import create_engine
# from sqlalchemy.ext.declarative import declarative_base
import psycopg2
import psycopg2.extras as extras
import pandas as pd
import os
import datetime
import numpy as np


def get_ohlc_stock_data(start_date, end_date, tickers):
    with psycopg2.connect("dbname=finrl user=dircks") as conn:
        with conn.cursor() as cursor:
            query = """
                select date, ticker, open, high, low, close, volume
                from data.stock_daily sd
                join data.stock s
                    on s.id = sd.stock_id
                where s.ticker in %(tickers)s and date between %(start)s and %(end)s
            """
            params = {
                'tickers': tuple(tickers),
                'start': start_date,
                'end': end_date
            }
            df = pd.read_sql(query, conn, 'date', params=params)
            return df
            # cursor.execute(query, (tuple(tickers), start_date, end_date))
            # return cursor.fetchall()


def delete_data_between_dates(stock_ids, start_date, end_date):
    with psycopg2.connect("dbname=finrl user=dircks") as conn:
        with conn.cursor() as cursor:
            query = """
                    delete from data.stock_daily 
                    where stock_id in %s and date between %s and %s;
                """

            cursor.execute(query, (tuple(stock_ids), start_date, end_date))


def save_single_stock(ticker):
    with psycopg2.connect("dbname=finrl user=dircks") as conn:
        with conn.cursor() as cursor:
            query = """
                insert into data.stock (ticker) values (%s) returning id;
            """
            cursor.execute(query, (ticker,))
            return cursor.fetchone()[0]


def save_stock_data(df):
    with psycopg2.connect("dbname=finrl user=dircks") as conn:
        with conn.cursor() as cursor:
            cursor.execute('select ticker, id from data.stock;')
            res = cursor.fetchall()

            df_tickers = df['tic'].unique()
            db_tickers = dict(res)
            existing_tickers = {ticker: db_tickers[ticker] for ticker in df_tickers if ticker in db_tickers}
            new_tickers = [ticker for ticker in df_tickers if ticker not in existing_tickers.keys()]

            # if we don't have the ticker in the db, add it
            for ticker in new_tickers:
                id = save_single_stock(ticker)
                existing_tickers[ticker] = id

            # delete existing data for these dates
            min_date = datetime.datetime.strptime(df['date'].min(), '%Y-%m-%d')
            max_date = datetime.datetime.strptime(df['date'].max(), '%Y-%m-%d')
            delete_data_between_dates(existing_tickers.values(), min_date, max_date)

            df['stock_id'] = df.apply(lambda row: existing_tickers[row.tic], axis=1)
            df = df[['stock_id', 'open', 'high', 'low', 'close', 'volume', 'date']]

            # bulk insert
            tuples = [tuple(row) for row in df.to_numpy()]
            cols = tuple(df.columns)

            query = """
                insert into data.stock_daily (stock_id, open, high, low, close, volume, date) values %s;
            """
            extras.execute_values(cursor, query, tuples)


if __name__ == "__main__":
    # df = pd.read_csv(os.path.abspath('../datasets/2004_2021_djia.csv'), index_col=0)
    # save_stock_data(df)
    print(get_ohlc_stock_data('2019-01-01', '2020-01-01', ['AAPL', 'AMZN']))
