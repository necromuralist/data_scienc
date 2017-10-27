# python standard library
import os
import pickle

# pypi
import networkx
import pandas
import seaborn

from numba import jit

from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    )
from sklearn.feature_selection import (
    RFECV,
    SelectFromModel,
    )
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    )
from sklearn.preprocessing import StandardScaler

class Futures(object):
    target = "Future Connection"
    data_file = "Future_Connections.csv"
    graph_file = "email_prediction.txt"
    networkx_data_index = 2
    folds = 10

class DataNames(object):
    resource_allocation = 'resource_allocation'
    jaccard = 'jaccard_coefficient'
    adamic = "adamic_adar"
    preferential = "preferential_attachment"

class Files(object):
    """File-names for data persistence"""
    future_training_data = 'future_training_data.csv'
    future_selection_outcomes = 'future_selection_outcomes.pkl'
    future_model_selection = "future_model_cvs.pkl"

class Training(object):
    """data-pickles"""
    x_train_lr_rfs = "x_train_lr_rfs.pkl"
    x_test_lr_rfs = "x_test_lr_rfs.pkl"
    x_train_trees_rfs = "x_train_trees_rfs.pkl"
    x_test_trees_rfs = "x_test_trees_rfs.pkl"
    x_train_lr_sfm = "x_train_lr_sfm.pkl"
    x_test_lr_sfm = "x_test_lr_sfm.pkl"
    x_train_trees_sfm = "x_train_trees_sfm.pkl"
    x_test_trees_sfm = "x_test_trees_sfm.pkl"

email = networkx.read_gpickle(Futures.graph_file)

future_connections_pre_loaded = os.path.isfile(Files.future_training_data)
if future_connections_pre_loaded:
    future_connections = pandas.read_csv(Files.future_training_data,
                                         index_col=0)
else:
    future_connections = pandas.read_csv(Futures.data_file,
                                         index_col=0,
                                         converters={0: eval})

def add_networkx_data(adder, name, graph=email, frame=future_connections):
    """Adds networkx data to the frame

    The networkx link-prediction functions return generators of triples:
     (first-node, second-node, value)

    This will use the index of the frame that's passed in as the source of 
    node-pairs for the networkx function (called `ebunch` in the networkx
    documentation) and the add only the value we want back to the frame

    Args:
     adder: networkx function to call to get the new data
     name: column-name to add to the frame
     graph: networkx graph to pass to the function
     frame (pandas.DataFrame): frame with node-pairs as index to add data to
    """
    frame[name] = [output[Futures.networkx_data_index]
                   for output in adder(graph, frame.index)]
    return frame

if not future_connections_pre_loaded:
    add_networkx_data(networkx.resource_allocation_index,
                      DataNames.resource_allocation)

if not future_connections_pre_loaded:
    add_networkx_data(networkx.jaccard_coefficient, DataNames.jaccard)

if not future_connections_pre_loaded:
    add_networkx_data(networkx.adamic_adar_index, DataNames.adamic)

if not future_connections_pre_loaded:
    add_networkx_data(networkx.preferential_attachment, DataNames.preferential)

future_connections.to_csv(Files.future_training_data)

prediction_set = future_connections[future_connections[Futures.target].isnull()]
training_set = future_connections[future_connections[Futures.target].notnull()]

non_target = [column for column in future_connections.columns
              if column != Futures.target]
training = training_set[non_target]
testing = training_set[Futures.target]
predictions = prediction_set[non_target]

x_train, x_test, y_train, y_test = train_test_split(training, testing, stratify=testing)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = pandas.DataFrame(x_train, columns=training.columns)
x_test = pandas.DataFrame(x_test, columns=training.columns)

def pickle_it(thing, name):
    """saves the thing as a pickle"""
    with open(name, "wb") as writer:
        pickle.dump(thing, writer)

def unpickle_it(name):
    """loads the object from the file-name

    Args:
     name (str): name of binary pickle file

    Returns:
     obj: unpickled object
    """
    with open(name, 'rb') as reader:
        thing = pickle.load(reader)
    return thing

if os.path.isfile(Training.x_train_lr_rfs):
    x_train_lr_rfs = unpickle_it(Training.x_train_lr_rfs)
    x_test_lr_rfs = unpickle_it(Training.x_test_lr_rfs)
else:
    estimator = LogisticRegressionCV(n_jobs=-1)
    selector = RFECV(estimator, scoring='roc_auc',
                     n_jobs=-1,
                     cv=StratifiedKFold(Futures.folds))
    x_train_lr_rfs = selector.fit_transform(x_train, y_train)
    x_test_lr_rfs = selector.transform(x_test)
    pickle_it(x_train_lr_rfs, Training.x_train_lr_rfs)
    pickle_it(x_test_lr_rfs, Training.x_test_lr_rfs)

if os.path.isfile(Training.x_train_trees_rfs):
    x_train_trees_rfs = unpickle_it(Training.x_train_trees_rfs)
    x_test_trees_rfs = unpickle_it(Training.x_test_trees_rfs)
else:
    estimator = ExtraTreesClassifier()
    selector = RFECV(estimator, scoring='roc_auc', n_jobs=-1, cv=StratifiedKFold(Futures.folds))
    x_train_trees_rfs = selector.fit_transform(x_train, y_train)
    x_test_trees_rfs = selector.transform(x_test)
    pickle_it(x_train_trees_rfs, Training.x_train_trees_rfs)
    pickle_it(x_test_trees_rfs, Training.x_test_trees_rfs)

if os.path.isfile(Training.x_train_lr_sfm):
    x_train_lr_sfm = unpickle_it(Training.x_train_lr_sfm)
    x_test_lr_sfm = unpickle_it(Training.x_test_lr_sfm)
else:
    estimator = LogisticRegressionCV(
        n_jobs=-1, scoring='roc_auc',
        cv=StratifiedKFold(Futures.folds)).fit(x_train,
                                               y_train)
    selector = SelectFromModel(estimator, prefit=True)
    x_train_lr_sfm = selector.transform(x_train)
    x_test_lr_sfm = selector.transform(x_test)
    pickle_it(x_train_lr_sfm, Training.x_train_lr_sfm)
    pickle_it(x_test_lr_sfm, Training.x_test_lr_sfm)

if os.path.isfile(Training.x_train_trees_sfm):
    x_train_trees_sfm = unpickle_it(Training.x_train_trees_sfm)
    x_test_trees_sfm = unpickle_it(Training.x_test_trees_sfm)
else:
    estimator = ExtraTreesClassifier()
    estimator.fit(x_train, y_train)
    selector = SelectFromModel(estimator, prefit=True)
    x_train_trees_sfm = selector.transform(x_train)
    x_test_trees_sfm = selector.transform(x_test)
    pickle_it(x_train_trees_sfm, Training.x_train_trees_sfm)
    pickle_it(x_test_trees_sfm, Training.x_test_trees_sfm)

if os.path.isfile(Files.future_model_selection):
    with open(Files.future_model_selection, 'rb') as pkl:
        scores = pickle.load(pkl)
else:
    scores = {}

def fit_and_print(estimator, x_train, x_test):
    """fits the estimator to the data

    Args:
     estimator: model to fit
     x_train: scaled data to fit model to
     x_test: data to test the model with

    Returns:
     tuple: model fit to the data, test score
    """
    model = estimator.fit(x_train, y_train)
    test_score = model.score(x_test, y_test)
    print("Mean Cross-Validation Score: {:.2f}".format(model.scores_[1].mean()))
    print("Testing Score: {:.2f}".format(test_score))
    return model, test_score

data_sets = {("extra trees", 'select from model') : (x_train_trees_sfm, x_test_trees_sfm),
             ("extra trees", 'recursive feature selection') : (x_train_trees_rfs, x_test_trees_rfs),
             ('logistic regression', "recursive feature selection") : (x_train_lr_rfs, x_test_lr_rfs),
             ('logistic regression', "select from model") : (x_train_lr_sfm, x_test_lr_sfm)}

def key_by_value(source, search_value):
    """Find the key in a dict that matches a value
    
    Args:
     source (dict): dictionary with value to search for
     search_value: value to search for

    Returns:
     object: key in source that matched value
    """
    for key, value in source.items():
        if value == search_value:
            return key
    return

def fit_and_print_all(model, model_name):
    """Fits the model against all data instances

    Args:
     model: model to fit to the data sets
     model_name: identifier for the outcomes
    """
    for data_set, x in data_sets.items():
        selector, method = data_set
        train, test = x
        key = ','.join([model_name, selector, method])
        print("Training Shape: {}".format(train.shape))
        if key not in scores:
            print(key)
            fitted, score = fit_and_print(model, train, test)
            scores[key] = score
        else:
            score = scores[key]
            print("{}: {:.3f}".format(key, score))
        print()

    best_score = max(scores.values())
    best_key = key_by_value(scores, best_score)
    print("Best Model So Far: {}, Score={:.2f}".format(
        best_key,
        best_score))
    with open(Files.future_model_selection, 'wb') as writer:
        pickle.dump(scores, writer)
    return

logistic_model = LogisticRegressionCV(n_jobs=-1, scoring="roc_auc",
                                      solver='liblinear',
                                      cv=StratifiedKFold(Futures.folds))
fit_and_print_all(logistic_model, "Logistic Regression")

def fit_grid_search(estimator, parameters, x_train, x_test):
    """Fits the estimator using grid search

    Args:
     estimator: Model to fit
     parameters (dict): hyper-parameters for the grid search
     x_train (array): the training data input
     x_test (array): data to evaluate the best model with

    Returns: 
     tuple: Best Model, best model score
    """
    search = GridSearchCV(estimator, parameters, n_jobs=-1, scoring='roc_auc',
                          cv=StratifiedKFold(Futures.folds))
    search.fit(x_train, y_train)
    best_model = search.best_estimator_
    test_score = best_model.score(x_test, y_test)
    print("Mean of Mean Cross-Validation Scores: {:.2f}".format(
        search.cv_results_["mean_train_score"].mean()))
    print("Mean of Cross-Validation Score STDs: {:.2f}".format(
        search.cv_results_["std_train_score"].mean()))
    print("Testing Score: {:.2f}".format(test_score))
    return best_model, test_score

def fit_grid_searches(estimator, parameters, name, data_sets=data_sets):
    """Fits the estimator against all the data-sets

    Args:
     estimator: instance of model to test
     parameters: dict of grid-search parameters
     name: identifier for the model
    """
    for data_set, x in data_sets.items():
        selector, method = data_set
        train, test = x
        key = ",".join([name, selector, method])
        if key not in scores:
            print(key)
            fitted, score = fit_grid_search(estimator, parameters, train, test)
            scores[key] = score
        else:
            score = scores[key]
            print("{}: {:.2f}".format(key, score))
        print()
    import pudb; pudb.set_trace()
    best = max(scores.items())
    best_key = key_by_value(scores, best)
    print("Best Model So Far: {}, Score={:.2f}".format(best_key, best))
    with open(Files.future_model_selection, 'wb') as writer:
        pickle.dump(scores, writer)
    return

parameters = dict(n_estimators = list(range(10, 20, 10)))
forest = RandomForestClassifier()
first = {("extra trees", 'select from model'): data_sets[("extra trees", 'select from model')]}
fit_grid_searches(forest, parameters, "Random Forest", first)

parameters = dict(n_estimators = list(range(10, 200, 10)))
trees = ExtraTreesClassifier()
fit_grid_searches(forest, parameters, "Extra Trees")
