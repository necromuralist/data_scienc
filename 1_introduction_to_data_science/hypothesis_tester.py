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
UNIVERSITY_TOWNS_DATA = "university_towns.txt"
GDP_DATA = "gdplev.xls"
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
class GDPSettings(object):
    """holder of settings to load and clean the data"""
    source = GDP_DATA
    skip_rows = 8
    columns = ["Year", "Annual GDP Current Billions",
               "Annual GDP 2009 Billions", "to_delete", "YearQuarter",
               "Quarterly GDP Current Billions", "Quarterly GDP 2009 Billions",
               "to_delete"]
    quarterly_column = "Quarterly GDP 2009 Billions"
    first_quarter = "2000q1"
    delete_columns = "to_delete"
class GDPData(BaseData):
    """GDP Data Loader and cleaner"""
    @property
    def data(self):
        if self._data is None:
            self._data = pandas.read_excel(self.settings.source,
                                           skiprows=self.settings.skip_rows,
                                           names=self.settings.columns)
            self._data = self._data.drop(self.settings.delete_columns,
                                         axis=COLUMN_AXIS)
        return self._data
def get_later_data():
    """
    Returns
    -------

    DataFrame: GDP data from 2000 q1
    """
    data = GDPData(GDPSettings).data
    data = data.iloc[data[data.YearQuarter==GDPSettings.first_quarter].index[0]:]
    return data.dropna(COLUMN_AXIS).reset_index()
def recession_index(data, compare=operator.gt):
    """returns the index of the start of the recession

    Parameters
    ----------

    data: DataFrame
       GDP data to search

    compare: function
       compare quarters (change to < to find end)

    Returns
    -------

    int : iloc of start of first recession found
    """
    for index, gdp in enumerate(data[GDPSettings.quarterly_column]):
        next_gdp = data[GDPSettings.quarterly_column].iloc[index + 1]
        if (index != 0 and
            (compare(data[GDPSettings.quarterly_column].iloc[index - 1],
             gdp) and compare(gdp, next_gdp))):
            return index
        elif (compare(gdp, next_gdp) and 
              compare(next_gdp, data[GDPSettings.quarterly_column].iloc[index + 2])):
            return index + 1
def get_recession_start():
    '''Returns the year and quarter of the recession start time as a 
    string value in a format such as 2005q3'''
    data = get_later_data()
    index = recession_index(data)
    return data.YearQuarter.iloc[index]
if DEBUG:
    print(get_recession_start())
def get_recession_end():
    '''Returns the year and quarter of the recession end time as a 
    string value in a format such as 2005q3'''
    data = get_later_data()
    start = get_recession_start()
    data = data[data[data.YearQuarter==start].index[0]:]
    index = recession_index(data, operator.lt)
    return data.YearQuarter.iloc[index]

if DEBUG:
    print(get_recession_end())
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
def convert_housing_data_to_quarters():
    '''Converts the housing data to quarters and returns it as mean 
    values in a dataframe. This dataframe should be a dataframe with
    columns for 2000q1 through 2016q3, and should have a multi-index
    in the shape of ["State","RegionName"].

    Note: Quarters are defined in the assignment description, they are
    not arbitrary three month periods.

    The resulting dataframe should have 67 columns, and 10,730 rows.
    '''
    housing = HousingData(HousingSettings)
    return housing.data

class HypothesisTester(object):
    """t-tester comparing university and non-university towns"""
    def __init__(self):
        self._start = None
        self._end = None
        self._data = None
        self._difference = None
        return

    @property
    def start(self):
        """start of the recession

        Returns
        -------

        String: column name of the start of the recession
        """
        if self._start is None:
            self._start = get_recession_start()
        return self._start

    @property
    def end(self):
        """end of the recession

        Returns
        -------

        String: column name of the end of the recession
        """
        if self._end is None:
            self._end = get_recession_end()
        return self._end

    @property
    def data(self):
        """housing data

        Returns
        -------

        DataFrame: housing data with quarter-means
        """
        if self._data is None:
            self._data = convert_housing_data_to_quarters()
        return self._data

    @property
    def difference(self):
        """difference between the end and start of the recession

        Returns
        -------

        DataFrame: difference
        """
        if self._difference is None:
            self._difference = self.data[self.end] - self.data[self.start]
        return self._difference
def run_ttest():
    '''First creates new data showing the decline or growth of housing prices
    between the recession start and the recession bottom. Then runs a ttest
    comparing the university town values to the non-university towns values, 
    return whether the alternative hypothesis (that the two groups are the same)
    is true or not as well as the p-value of the confidence. 

    Return the tuple (different, p, better) where different=True if the t-test is
    True at a p<0.01 (we reject the null hypothesis), or different=False if 
    otherwise (we cannot reject the null hypothesis). The variable p should
    be equal to the exact p value returned from scipy.stats.ttest_ind(). The
    value for better should be either "university town" or "non-university town"
    depending on which has a lower mean price ratio (which is equivilent to a
    reduced market loss).'''
    return "ANSWER"
tester = HypothesisTester()
