# python standard library
from abc import ABCMeta
from abc import abstractproperty
import operator
import re

# third party
import matplotlib
import numpy 
import pandas
from scipy.stats import ttest_ind  

COLUMN_AXIS = 1 
DEBUG = True

class BaseData(object):
    """Base class for the data-loaders

    Parameters
    ----------

    settings: object
       holder of settings needed to load the data
    """
    __meta__ = ABCMeta
    def __init__(self, settings):
        self.settings = settings
        self._data = None
        return

    @abstractproperty
    def data(self):
        """the data-frame"""
        return
class HousingSettings(object):
    """holds the settings for the housing data"""
    source = "City_Zhvi_AllHomes.csv"
    years = ["20{0:02d}".format(year) for year in range(17)]
    year_month_pattern = re.compile("20\d\d-\d\d")
    quarters = [re.compile("|".join(["{0:02d}".format(month) for month in range(start, start+3)])) for start in range(1, 11, 3)]

class HousingData(BaseData):
    """loads the housing data"""
    @property
    def data(self):
        """
        Returns
        -------

        DataFrame: frame with the zillow housing data
        """
        if self._data is None:
            self._data = pandas.read_csv(self.settings.source)
            quarters = self.convert_quarters(self._data)
            tuples = [self._data.State, self._data.RegionName]
            multi_index = pandas.MultiIndex.from_tuples(list(zip(*tuples)),
                                                        names=["State", "RegionName"])
            self._data = quarters.set_index(multi_index)
        return self._data

    def convert_quarters(self, data):
        """creates a data-frame with the data as means of quarters

        Parameters
        ----------

        data: DataFrame
           Housing Data

        Returns
        -------

        DataFrame: month columns from data converted to means of quarters
        """
        all_years = data.select(lambda x: self.settings.year_month_pattern.match(x),
                               axis=COLUMN_AXIS)
        means = {}
        for year_label in self.settings.years:
            year = all_years.select(lambda x: re.search(year_label, x),
                                    axis = COLUMN_AXIS)
            for index, quarter_regex in enumerate(self.settings.quarters):
                quarter = year.select(lambda x: quarter_regex.search(x),
                                      axis=COLUMN_AXIS)
                means["{0}q{1}".format(year_label, index+1)] = quarter.mean(axis=COLUMN_AXIS)
        return pandas.DataFrame(means).dropna(axis="columns", how="all")

# housing = HousingData(HousingSettings)
# converted = housing.convert_quarters(housing.data)
