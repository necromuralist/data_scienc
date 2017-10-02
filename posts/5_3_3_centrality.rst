.. title: Measures of Centrality
.. slug: measures-of-centrality
.. date: 2017-10-01 16:42
.. tags: networks, centrality
.. link: 
.. description: A look at measures of node centrality
.. type: text
.. author: hades

`Node Centrality <https://en.wikipedia.org/wiki/Centrality>`_ is a measure of the importance of a node to a network. This will explore measures of centrality using two networks, a friendship network, and a blog network.

1 Part 1 - Friendships
----------------------

This will look at a network of friendships at a university department. Each node corresponds to a person (identified by an integer node label), and an edge indicates friendship. 

1.1 Imports
~~~~~~~~~~~

.. code:: ipython

    import networkx

1.2 Friendships data
~~~~~~~~~~~~~~~~~~~~

.. code:: ipython

    friendships = networkx.read_gml('friendships.gml')

.. code:: ipython

    print(len(friendships))
    print(networkx.is_connected(friendships))
    print(networkx.is_directed(friendships))

::

    1133
    True
    False

There are 1,133 people in the friendship network, which is a connected, undirected graph.

1.3 Degree, Closeness, and Normalized Betweenness Centrality
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Find the degree centrality, closeness centrality, and normalized betweenness centrality (excluding endpoints) of node 100.

- `Degree Centrality <https://en.wikipedia.org/wiki/Centrality#Degree_centrality>`_ scores the nodes based on the number of links they have to other nose. The assumption is that a node with more connections should be more important.

- `Closeness Centrality <https://en.wikipedia.org/wiki/Closeness_centrality>`_ uses the lengths of shortest paths to decide importance. The less distance there is between a node and the other nodes the more important it is.

- `Betweenness Centrality <https://en.wikipedia.org/wiki/Betweenness_centrality>`_ counts the number of shortest paths between pairs of nodes that pass through a node.

.. code:: ipython

    DEGREE_CENTRALITY = networkx.degree_centrality(friendships)
    CLOSENESS_CENTRALITY = networkx.closeness_centrality(friendships)
    BETWEENNESS_CENTRALITY = networkx.betweenness_centrality(friendships)

.. code:: ipython

    def node_centrality(node=100):
        """gets measures of centrality for node

        Args:
         node (int): the number (key) for the node

        Returns:
         tuple: 
          - float: degree centrality
          - float: closeness centrality
          - float: normalized betweeness centrality
        """
        return (DEGREE_CENTRALITY[node],
                CLOSENESS_CENTRALITY[node], BETWEENNESS_CENTRALITY[node])

.. code:: ipython

    print("Node 101:")
    degree, closeness, betweenness = node_centrality()
    print("Degree Centrality: {0:.4f}".format(degree))
    print("Closeness Centrality: {0:.2f}".format(closeness))
    print("Normalized Betweenness Centrality: {0:.6f}".format(betweenness))

::

    Node 101:
    Degree Centrality: 0.0027
    Closeness Centrality: 0.27
    Normalized Betweenness Centrality: 0.000071


.. code:: ipython

    def largest_node(centrality):
        """gets the node with the best (largest) score

        Args:
         centrality (dict): one of the centrality score dicts

        Returns:
         int: name of the node with the best score
        """
        return list(
            reversed(sorted((value, node)
                            for (node, value) in centrality.items())))[0][1]

1.4 Most Connected Friend
~~~~~~~~~~~~~~~~~~~~~~~~~

We want to contact one person in our friendship network and have him or her contact all his or her immediate friends. To have the greatest impact, this person should have the most links in the network. Which node is this?

.. code:: ipython

    def most_connected_friend():
        """returns the node with the best degree centrality"""
        return largest_node(DEGREE_CENTRALITY)

.. code:: ipython

    MOST_CONNECTED = most_connected_friend()
    print("Most Connected Friend: {}".format(MOST_CONNECTED))

::

    Most Connected Friend: 105

.. code:: ipython

    connected = networkx.Graph()
    friends = friendships[MOST_CONNECTED]
    for friend in friends:
        connected.add_edge(MOST_CONNECTED, friend)
    positions = networkx.spring_layout(connected)
    networkx.draw(connected, positions, with_labels=False, node_color='b', node_size=50)
    networkx.draw(connected, positions, nodelist=[MOST_CONNECTED], node_color="r")

.. image:: most_connected_friend.png

Node 105 does appear to be well connected.

1.5 Fewest Hops
~~~~~~~~~~~~~~~

We want to reach everyone in the network by having one person passing messages to his friends who can then pass it on and so forth (a six-degrees of separation type scenario) but we want the fewest number of transfers. *Which friend is closest to all the people in the friendship network?*

.. code:: ipython

    def closest_friend():
        """the node with the best closeness centrality

        Returns:
         int: Identifier for the node closest to all the other nodes
        """
        return largest_node(CLOSENESS_CENTRALITY)

.. code:: ipython

    CLOSEST_FRIEND = closest_friend()
    print("Closest Friend: {}".format(CLOSEST_FRIEND))

::

    Closest Friend: 23

.. code:: ipython

    positions = networkx.spring_layout(friendships)
    networkx.draw(friendships, positions, node_size=1, alpha=0.25, node_color='b')
    networkx.draw_networkx_nodes(friendships, positions, nodelist=[CLOSEST_FRIEND],
                                 node_color='r', node_size=50)

.. image:: closest_friend.png

Interesting to look at, if not the most informative.

1.6 Most Important Connection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Although the graph is connected, if you took out one persion from the network, which one would cause the most disruption (which person is in the path of the most shortest paths)?

.. code:: ipython

    def betweenness_centrality():
        """the node with the highest betweenness centrality

        Returns:
         int: ID of the person who sits on the most shortest paths
        """
        return largest_node(BETWEENNESS_CENTRALITY)

.. code:: ipython

    MOST_BETWEEN = betweenness_centrality()
    print("Most Between Friend: {}".format(MOST_BETWEEN))

::

    Most Between Friend: 333

Node 333 sits on the most shortest paths between pairs of nodes.    

2 Part 2 - Political Blogs
--------------------------

Now we're going to use `PageRank <https://en.wikipedia.org/wiki/PageRank>`_ and `Hyperlink-Induced Topic Search (HITS) <https://en.wikipedia.org/wiki/HITS_algorithm>`_  to look at a directed network of political blogs, where nodes correspond to a blog and edges correspond to links between blogs.

.. code:: ipython

    blogs = networkx.read_gml('blogs.gml')

.. code:: ipython

    print(len(blogs))
    print(networkx.is_directed(blogs))

::

    1490
    True

.. code:: ipython

    networkx.draw(blogs, alpha=0.5, node_size=1, node_color='r')

.. image:: political.png

2.1 Scaled Page Rank of *realclearpolitics.com*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*PageRank* scores web-pages by the number of important nodes that link directly to them. It is possible for the algorithm to get stuck if there are no edges leading out from a directed subgraph, producing erroneous page-ranks so the *Scaled Page Rank* uses a random-restart do decide when to occasionally jump to a new node, an idea similar to the way Stochastic Gradient Descent avoids being stuck in local minima. The `Networkx pagerank <https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html>`_ uses a default of 0.85, which I will use, so it will do a random-restart about 15% of the time.

.. code:: ipython

    PAGE_RANK = networkx.pagerank(blogs)

.. code:: ipython

    def real_clear_politics_page_rank():
        """Page Rank of realclearpolitics.com

        Returns:
         float: The PageRank for the realclearpolitics blog.
        """
        return PAGE_RANK['realclearpolitics.com']

.. code:: ipython

    print("Real Clear Politics Page Rank: {0:.4f}".format(real_clear_politics_page_rank()))

::

    Real Clear Politics Page Rank: 0.0046

2.2 Top Five Blogs by Page Rank
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This time the PageRank scores will be used to find what it thinks are the most important blogs.

.. code:: ipython

    def top_five(ranks, count=5):
        """gets the top-five blogs by rank

        Args:
         count (int): number to return

        Returns:
         list [str]: names of the top blogs - most to least important
        """
        top = list(reversed(sorted((rank, node)
                                   for node, rank in ranks.items())))[:count]
        return [node for rank, node in top]

.. code:: ipython

    def top_five_page_rank():
        """Top 5 nodes by page rank

        Returns:
         list [str]: top-five blogs by page-rank
        """
        return top_five(PAGE_RANK)

.. code:: ipython

    print("Top Five Blogs by PageRank")

    for blog in top_five_page_rank():
        print("  - {}".format(blog))

::

    Top Five Blogs by PageRank
      - dailykos.com
      - atrios.blogspot.com
      - instapundit.com
      - blogsforbush.com
      - talkingpointsmemo.com

The fact that `dailykos`, `instapundit`, and `talkingpointsmemo` are there doesn't seem too surprising. The fact that a blogspot blog and `blogsforbush` are there are a little surprising. I assume the data-set is old (Blogs for Bush is a wordpress blog whose last post was in 2004).  `atrios.blogspot.com` currently redirects to `www.eschatonblog.com`, and looks like it was made in the 1990s, but the posts are recent.

2.3 HITS Score for Real Clear Politics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This uses the HITS algorithm to find the authority and hub scores for *realclearpolitics.com*. This algorithm tries to identify ``hubs``, collections of links that directed users to important pages, and ``authoratative`` pages, pages that are deemed important because of their relevant content (as identified by the fact that they are linked to by ``hubs``).

.. code:: ipython

    HUBS, AUTHORITIES = networkx.hits(blogs)

.. code:: ipython

    def real_clear_politics_hits():
        """HITS score for realclearpolitics.com

        Returns:
         tuple (float, float): hub score, authority score
        """
        return HUBS['realclearpolitics.com'], AUTHORITIES['realclearpolitics.com']

.. code:: ipython

    hub, authority = real_clear_politics_hits()
    print("Real Clear Politics")
    print("Hub: {0:.5f}\nAuthority: {0:.5f}".format(hub, authority))

::

    Real Clear Politics
    Hub: 0.00032
    Authority: 0.00032

2.4 Top 5 Blogs by Hub Score
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This will find the top five blogs based on their hub scores (meaning they are the ones who link to the most authoratative sites).

.. code:: ipython

    def top_five_hubs():
        """Top five blogs by hub scores

        Returns:
         list (str): Names of top-five hub blogs
        """
        return top_five(HUBS)

.. code:: ipython

    top_five_hub_blogs = top_five_hubs()
    print('Top Five Hub Blogs')
    for blog in top_five_hub_blogs:
        print(" - {}".format(blog))

::

    Top Five Hub Blogs
     - politicalstrategy.org
     - madkane.com/notable.html
     - liberaloasis.com
     - stagefour.typepad.com/commonprejudice
     - bodyandsoul.typepad.com

2.5 Top Five Blogs By Authority
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This will find the top five political blogs based on how many of the hub-blogs link to them.

.. code:: ipython

    def top_five_authorities():
        """the top 5 blogs by authorities score

        Returns:
         list (str): names of the most authoratative blogs
        """
        return top_five(AUTHORITIES)

.. code:: ipython

    print("Top Five Authoratative Blogs")
    authoratative_blogs = top_five_authorities()
    for blog in authoratative_blogs:
        print(" - {}".format(blog))

::

    Top Five Authoratative Blogs
     - dailykos.com
     - talkingpointsmemo.com
     - atrios.blogspot.com
     - washingtonmonthly.com
     - talkleft.com
