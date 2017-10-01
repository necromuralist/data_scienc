.. title: Measures of Centrality
.. slug: Measures-of-Centrality
.. date: 2017-09-30 17:56
.. tags: networks,centrality
.. link: 
.. description: Centrality measurements for a friendship network and political blogs
.. type: text
.. author: hades

In this assignment you will explore measures of centrality on two networks, a friendship network in Part 1, and a blog network in Part 2.

Part 1 - Friendships
--------------------

1.1 Imports
~~~~~~~~~~~

.. code:: ipython

    import networkx

1.2 Friendships data
~~~~~~~~~~~~~~~~~~~~

.. code:: ipython

    friendships = networkx.read_gml('friendships.gml')

1.3 Question 1
~~~~~~~~~~~~~~

Find the degree centrality, closeness centrality, and normalized betweeness centrality (excluding endpoints) of node 100.

**This function should return a tuple of floats ``(degree_centrality, closeness_centrality, betweenness_centrality)``.**

.. code:: ipython

    DEGREE_CENTRALITY = networkx.degree_centrality(friendships)
    CLOSENESS_CENTRALITY = networkx.closeness_centrality(friendships)
    BETWEENNESS_CENTRALITY = networkx.betweenness_centrality(friendships)

.. code:: ipython

    def answer_one():
        """gets measures of centrality for node 100

        Returns:
         tuple: 
          - float: degree centrality
          - float: closeness centrality
          - float: normalized betweeness centrality
        """
        NODE = 100
        return (DEGREE_CENTRALITY[NODE],
                CLOSENESS_CENTRALITY[NODE], BETWEENNESS_CENTRALITY[NODE])

.. code:: ipython

    print(answer_one())

::

    (0.0026501766784452294, 0.2654784240150094, 7.142902633244772e-05)


For Questions 2, 3, and 4, use one of the covered centrality measures to rank the nodes and find the most appropriate candidate.

.. code:: ipython

    def largest_node(centrality):
        """gets the node with the best (largest score)

        Returns:
         int: name of the node with the best score
        """
        return list(reversed(sorted((value, node)
                                    for (node, value) in centrality.items())))[0][1]

1.4 Question 2
~~~~~~~~~~~~~~

Suppose you are employed by an online shopping website and are tasked with selecting one user in network G1 to send an online shopping voucher to. We expect that the user who receives the voucher will send it to their friends in the network.  You want the voucher to reach as many nodes as possible. The voucher can be forwarded to multiple users at the same time, but the travel distance of the voucher is limited to one step, which means if the voucher travels more than one step in this network, it is no longer valid. Apply your knowledge in network centrality to select the best candidate for the voucher. 

**This function should return an integer, the name of the node.**

.. code:: ipython

    def answer_two():
        """returns the node with the best degree centrality"""
        return largest_node(DEGREE_CENTRALITY)

.. code:: ipython

    print(answer_two())

::

    105

1.5 Question 3
~~~~~~~~~~~~~~

Now the limit of the voucher’s travel distance has been removed. Because the network is connected, regardless of who you pick, every node in the network will eventually receive the voucher. However, we now want to ensure that the voucher reaches the nodes in the lowest average number of hops.

How would you change your selection strategy? Write a function to tell us who is the best candidate in the network under this condition.

**This function should return an integer, the name of the node.**

.. code:: ipython

    def answer_three():
        """Returns the node with the best closeness centrality"""
        return largest_node(CLOSENESS_CENTRALITY)

.. code:: ipython

    print(answer_three())

::

    23

1.6 Question 4
~~~~~~~~~~~~~~

Assume the restriction on the voucher’s travel distance is still removed, but now a competitor has developed a strategy to remove a person from the network in order to disrupt the distribution of your company’s voucher. Identify the single riskiest person to be removed under your competitor’s strategy?

**This function should return an integer, the name of the node.**

.. code:: ipython

    def answer_four():
        """the node with the highest betweenness centrality"""
        return largest_node(BETWEENNESS_CENTRALITY)

.. code:: ipython

    print(answer_four())

::

    333

Part 2 - Political Blogs
------------------------

``blogs`` is a directed network of political blogs, where nodes correspond to a blog and edges correspond to links between blogs. Use your knowledge of PageRank and HITS to answer Questions 5-9.

.. code:: ipython

    blogs = networkx.read_gml('blogs.gml')

2.1 Question 5
~~~~~~~~~~~~~~

Apply the Scaled Page Rank Algorithm to this network. Find the Page Rank of node 'realclearpolitics.com' with damping value 0.85.

**This function should return a float.**

.. code:: ipython

    PAGE_RANK = networkx.pagerank(blogs)

.. code:: ipython

    def answer_five():
        """Page Rank of realclearpolitics.com"""
        return PAGE_RANK['realclearpolitics.com']

.. code:: ipython

    print(answer_five())

::

    0.004636694781649093

2.2 Question 6
~~~~~~~~~~~~~~

Apply the Scaled Page Rank Algorithm to this network with damping value 0.85. Find the 5 nodes with highest Page Rank. 

**This function should return a list of the top 5 blogs in desending order of Page Rank.**

.. code:: ipython

    def top_five(ranks):
        """gets the top-five blogs by rank"""
        top = list(reversed(sorted((rank, node)
                                   for node, rank in ranks.items())))[:5]
        return [node for rank, node in top]

.. code:: ipython

    def answer_six():
        """Top 5 nodes by page rank"""
        return top_five(PAGE_RANK)

.. code:: ipython

    print(answer_six())

::

    ['dailykos.com', 'atrios.blogspot.com', 'instapundit.com', 'blogsforbush.com', 'talkingpointsmemo.com']

2.3 Question 7
~~~~~~~~~~~~~~

Apply the HITS Algorithm to the network to find the hub and authority scores of node 'realclearpolitics.com'. 

**Your result should return a tuple of floats \`(hub\_score, authority\_score)\`.**

.. code:: ipython

    HUBS, AUTHORITIES = networkx.hits(blogs)

.. code:: ipython

    def answer_seven():
        """HITS score for realclearpolitics.com"""
        return HUBS['realclearpolitics.com'], AUTHORITIES['realclearpolitics.com']

.. code:: ipython

    print(answer_seven())

::

    (0.0003243556140916669, 0.003918957645699851)

2.4 Question 8
~~~~~~~~~~~~~~

Apply the HITS Algorithm to this network to find the 5 nodes with highest hub scores.

**This function should return a list of the top 5 blogs in desending order of hub scores.**

.. code:: ipython

    def answer_eight():
        """Top five blogs by hub scores"""
        return top_five(HUBS)

.. code:: ipython

    print(answer_eight())

::

    ['politicalstrategy.org', 'madkane.com/notable.html', 'liberaloasis.com', 'stagefour.typepad.com/commonprejudice', 'bodyandsoul.typepad.com']

Apply the HITS Algorithm to this network to find the 5 nodes with highest authority scores.

**This function should return a list of the top 5 blogs in desending order of authority scores.**

.. code:: ipython

    def answer_nine():
        """the top 5 blogs by authorities score"""
        return top_five(AUTHORITIES)

.. code:: ipython

    print(answer_nine())

::

    ['dailykos.com', 'talkingpointsmemo.com', 'atrios.blogspot.com', 'washingtonmonthly.com', 'talkleft.com']
