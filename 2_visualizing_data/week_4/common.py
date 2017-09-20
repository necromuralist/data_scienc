# python standard library
import os

# pypi
import pandas
import requests


class Urls(object):
    national = "https://www.dropbox.com/s/qw2l5hmu061l8x2/national_unemployment.csv?dl=1"
    portland = "https://www.dropbox.com/s/wvux3d7dcaae5t0/portland_unemployment_2007_2017.csv?dl=1"
    house_price = "https://www.dropbox.com/s/4hu2jpjkhcnr35k/purchase_only_house_price_index.csv?dl=1"
    s_and_p = "https://www.dropbox.com/s/ojj5zp7feid6wwl/SP500_index.csv?dl=1"


class Paths(object):
    portland = "portland_unemployment_2007_2017.csv"
    national = "national_unemployment.csv"
    s_and_p = "SP500_index.csv"
    house_price = "purchase_only_house_price_index.csv"


def download_data(path, url, na_values=None):
    """downloads the file if it doesn't exist
    Args:
     path (str): path to the file
     url (str): download url
     na_values (str|list): what pands will interpret as missing
    returns: DataFrame created from the file
    """
    if not os.path.isfile(path):
        response = requests.get(url)
        with open(path, 'w') as writer:
            writer.write(response.text)
    return pandas.read_csv(path, na_values=na_values)
