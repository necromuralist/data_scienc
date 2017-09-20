import networkx
import pandas
import seaborn

email_network = pandas.read_table('email_network.txt', dtype={"#Sender": str, "Recipient": str})
print(email_network.head())

def answer_one():
    """Loads the email-network graph

    Returns:
     networkx.MultiDiGraph: the graph of the email network
    """
    # there's a bug in networkx loading MultiDiGraphs from pandas data-frames
    # so this is a work-around
    graph = networkx.MultiDiGraph()
    tuples = [(sender, recipient, {"time": time})
              for (sender, recipient, time) in email_network.values]
    graph.add_edges_from(tuples)
    return graph
