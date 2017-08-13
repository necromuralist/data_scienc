.. title: Assignment 2 - Introduction to NLTK
.. slug: assignment-2-introduction-to-nltk
.. date: 2017-08-12 15:02
.. tags: text nltk nlp
.. link: 
.. description: Analyzing Moby Dick with the NLTK.
.. type: text
.. author: hades

In part 1 of this assignment you will use nltk to explore the Herman Melville novel Moby Dick. Then in part 2 you will create a spelling recommender function that uses nltk to find words similar to the misspelling. 

1 Part 1 - Analyzing Moby Dick
------------------------------

1.1 Imports and Data Set Up
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython

    from nltk.probability import FreqDist
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import sent_tokenize
    import matplotlib
    import matplotlib.pyplot as pyplot
    import nltk
    import nltk.data
    import numba
    import numpy
    import pandas
    import seaborn

.. code:: ipython

    %matplotlib inline
    seaborn.set_style("whitegrid")

If you would like to work with the raw text you can use 'moby\_raw'.

.. code:: ipython

    with open('moby.txt', 'r') as reader:
        moby_raw = reader.read()

If you would like to work with the novel in ``nltk.Text`` format you can use 'text1'.

.. code:: ipython

    moby_tokens = nltk.word_tokenize(moby_raw)
    text = nltk.Text(moby_tokens)
    moby_series = pandas.Series(moby_tokens)

1.2 Examples
~~~~~~~~~~~~

1.2.1 Example 1
^^^^^^^^^^^^^^^

*How many tokens (words and punctuation symbols) are in text1?* A **token** is a linguistic unit such as a word, punctuation mark, or alpha-numeric strings.

**This function should return an integer.**

.. code:: ipython

    def example_one():
         """counts the tokens in moby dick

         Returns:
          int: number of tokens in moby dick
         """
         # or alternatively len(text1)
         return len(moby_tokens)

    MOBY_TOKEN_COUNT = example_one()
    print("Moby Dick has {:,} tokens.".format(
         MOBY_TOKEN_COUNT))

::

    Moby Dick has 254,989 tokens.

1.2.2 Example 2
^^^^^^^^^^^^^^^

*How many unique tokens (unique words and punctuation) does text1 have?*

**This function should return an integer.**

.. code:: ipython

    def example_two():
        """counts the unique tokens

        Returns:
         int: count of unique tokens in Moby Dick
        """
        # or alternatively len(set(text1))
        return len(set(nltk.word_tokenize(moby_raw)))

    MOBY_UNIQUE_COUNT = example_two()
    print("Moby Dick has {:,} unique tokens.".format(
        MOBY_UNIQUE_COUNT))

::

    Moby Dick has 20,755 unique tokens.

1.2.3 Example 3
^^^^^^^^^^^^^^^

*After lemmatizing the verbs, how many unique tokens does text1 have?* A **lemma** is the canonical form. e.g. *run* is the lemma for *runs*, *ran*, *running*, and *run*.

**This function should return an integer.**

.. code:: ipython

    def example_three():
        """Counts the number of lemma in Moby Dick

        Returns:
         int: count of unique lemma
        """
        lemmatizer = WordNetLemmatizer()
        return len(set([lemmatizer.lemmatize(w,'v') for w in text1]))

    MOBY_LEMMA_COUNT = example_three()
    print("Moby Dick has {:,} lemma (found in WordNet).".format(
        MOBY_LEMMA_COUNT))

::

    Moby Dick has 16,900 lemma (found in WordNet).

1.3 Questions
~~~~~~~~~~~~~

1.3.1 Question 1
^^^^^^^^^^^^^^^^

What is the lexical diversity of the given text input? (i.e. ratio of unique tokens to the total number of tokens)

**This function should return a float.**

.. code:: ipython

    @jit
    def lexical_diversity(tokens):
        """Calculates the lexical diversity of a list of tokens
    
        Returns:
         float: fraction of tokens that are unique
        """    
        return len(set(tokens))/float(len(tokens))

.. code:: ipython

    def answer_one():
        """Calculates the lexical diversity of Moby Dick
    
        Returns:
         float: fraction of tokens that are unique
        """    
        return lexical_diversity(moby_tokens)

    output = answer_one()
    print("Lexical Diversity of Moby Dick: {:.2f}".format(output))

::

    Lexical Diversity of Moby Dick: 0.08

About 8 percent of the tokens in Moby Dick are unique.

1.3.2 Question 2
^^^^^^^^^^^^^^^^

*What percentage of tokens is 'whale'or 'Whale'?*

**This function should return a float.**

.. code:: ipython

    moby_frequencies = FreqDist(moby_tokens)

.. code:: ipython

    def answer_two():
        """calculates percentage of tokens that are 'whale'

        Returns:
         float: percentage of entries that are whales
        """
        whales = moby_frequencies["whale"] + moby_frequencies["Whale"]
        return 100 * (whales/float(MOBY_TOKEN_COUNT))

    whale_fraction = answer_two()
    print("Percentage of tokens that are whales: {:.2f} %".format(whale_fraction))

::

    Percentage of tokens that are whales: 0.41 %

Around 1 percent of the tokens are 'whale'.

I originally made two mistakes with this question, I was returning a fraction, not a percentage, and I was using a regular expression *'([Ww]hale)'* which I later realized would match *whales*, *whaler*, and other variants.


::

    782

1.3.3 Question 3
^^^^^^^^^^^^^^^^

*What are the 20 most frequently occurring (unique) tokens in the text? What is their frequency?*

**This function should return a list of 20 tuples where each tuple is of the form \`(token, frequency)\`. The list should be sorted in descending order of frequency.**


.. code:: ipython

    def answer_three():
        """finds 20 most requently occuring tokens

        Returns:
         list: (token, frequency) for top 20 tokens
        """
        return moby_frequencies.most_common(20)

    print(answer_three())

::

    [(',', 19204), ('the', 13715), ('.', 7308), ('of', 6513), ('and', 6010), ('a', 4545), ('to', 4515), (';', 4173), ('in', 3908), ('that', 2978), ('his', 2459), ('it', 2196), ('I', 2097), ('!', 1767), ('is', 1722), ('--', 1713), ('with', 1659), ('he', 1658), ('was', 1639), ('as', 1620)]

1.3.4 Question 4
^^^^^^^^^^^^^^^^

*What tokens have a length of greater than 5 and frequency of more than 150?*

**This function should return a sorted list of the tokens that match the above constraints. To sort your list, use \`sorted()\`**

.. code:: ipython

    moby_frequency_frame = pandas.DataFrame(moby_frequencies.most_common(),
                                            columns=["token", "frequency"])

.. code:: ipython

    def answer_four():
        """gets tokens with length > 5, frequency > 150"""
        frame =  moby_frequency_frame[(moby_frequency_frame.frequency > 150)
                                      & (moby_frequency_frame.token.str.len() > 5)]
        return sorted(frame.token)

    output = answer_four()
    print(output)

::

    ['Captain', 'Pequod', 'Queequeg', 'Starbuck', 'almost', 'before', 'himself', 'little', 'seemed', 'should', 'though', 'through', 'whales', 'without']

I was originally returning the data frame, not just the sorted tokens, which was of course marked wrong.

1.3.5 Question 5
^^^^^^^^^^^^^^^^

*Find the longest word in text1 and that word's length.*

**This function should return a tuple \`(longest\_word, length)\`.**

.. code:: ipython

    def answer_five():
        """finds the longest word and its length

        Return:
         tuple: (longest-word, length)
        """
        length = max(moby_frequency_frame.token.str.len())
        longest = moby_frequency_frame.token.str.extractall("(?P<long>.{{{}}})".format(length))
        return (longest.long.iloc[0], length)

    print(answer_five())

::

    ("twelve-o'clock-at-night", 23)

1.3.6 Question 6
^^^^^^^^^^^^^^^^

*What unique words have a frequency of more than 2000? What is their frequency?*

Hint:  you may want to use \`isalpha()\` to check if the token is a word and not punctuation.

**This function should return a list of tuples of the form \`(frequency, word)\` sorted in descending order of frequency.**

.. code:: ipython

    moby_words = moby_frequency_frame[moby_frequency_frame.token.str.isalpha()]

.. code:: ipython

    def answer_six():
        """Finds words wih frequency > 2000

        Returns:
         list: frequency, word tuples
        """
        common = moby_words[moby_words.frequency > 2000]
        return list(zip(common.frequency, common.token))

    print(answer_six())

::

    [(13715, 'the'), (6513, 'of'), (6010, 'and'), (4545, 'a'), (4515, 'to'), (3908, 'in'), (2978, 'that'), (2459, 'his'), (2196, 'it'), (2097, 'I')]

When I first submitted this I got it wrong because I was returning a list of ``(word, frequency)``, not ``(frequency, word)``.

1.3.7 Question 7
^^^^^^^^^^^^^^^^

*What is the average number of tokens per sentence?*

**This function should return a float.**

.. code:: ipython

    def answer_seven():
        """average number of tokens per sentence"""
        sentences = sent_tokenize(moby_raw)
        counts = (len(nltk.word_tokenize(sentence)) for sentence in sentences)
        return sum(counts)/float(len(sentences))

    output = answer_seven()
    print("Average number of tokens per sentence: {:.2f}".format(output))

::

    Average number of tokens per sentence: 25.88

1.3.8 Question 8
^^^^^^^^^^^^^^^^

*What are the 5 most frequent parts of speech in this text? What is their frequency?* Parts of Speech (POS) are the lexical categories that words belong to.

**This function should return a list of tuples of the form \`(part\_of\_speech, frequency)\` sorted in descending order of frequency.**

.. code:: ipython

    def answer_eight():
        """gets the 5 most frequent parts of speech

        Returns:
         list (Tuple): (part of speech, frequency) for top 5
        """
        tags = nltk.pos_tag(moby_words.token)
        frequencies = FreqDist([tag for (word, tag) in tags])
        return frequencies.most_common(5)

    output = answer_eight()
    print("Top 5 parts of speech: {}".format(output))

::

    Top 5 parts of speech: [('NN', 4016), ('NNP', 2916), ('JJ', 2875), ('NNS', 2452), ('VBD', 1421)]

2 Part 2 - Spelling Recommender
-------------------------------

For this part of the assignment you will create three different spelling recommenders, that each take a list of misspelled words and recommends a correctly spelled word for every word in the list.

For every misspelled word, the recommender should find find the word in \`correct\_spellings\` that has the shortest distance\*, and starts with the same letter as the misspelled word, and return that word as a recommendation.

**Each of the three different recommenders will use a different distance measure (outlined below)**.

Each of the recommenders should provide recommendations for the three default words provided: \`['cormulent', 'incendenece', 'validrate']\`.

.. code:: ipython

    from nltk.corpus import words
    from nltk.metrics.distance import (
        edit_distance,
        jaccard_distance,
        )
    from nltk.util import ngrams

.. code:: ipython

    correct_spellings = words.words()
    spellings_series = pandas.Series(correct_spellings)

2.1 Question 9
~~~~~~~~~~~~~~

For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:

`Jaccard distance <https://en.wikipedia.org/wiki/Jaccard_index>`_ on the trigrams of the two words.

**This function should return a list of length three:** ``['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']``.

.. code:: ipython

    def jaccard(entries, gram_number):
        """find the closet words to each entry

        Args:
         entries: collection of words to match
         gram_number: number of n-grams to use

        Returns:
         list: words with the closest jaccard distance to entries
        """
        outcomes = []
        for entry in entries:
            spellings = spellings_series[spellings_series.str.startswith(entry[0])]
            distances = ((jaccard_distance(set(ngrams(entry, gram_number)),
                                           set(ngrams(word, gram_number))), word)
                         for word in spellings)
            closest = min(distances)
            outcomes.append(closest[1])
        return outcomes

.. code:: ipython

    def answer_nine(entries=['cormulent', 'incendenece', 'validrate']):
        """finds the closest word based on jaccard distance"""
        return jaccard(entries, 3)
    
    print(answer_nine())

::

    ['corpulent', 'indecence', 'validate']

I originally got both the Jaccard Distance problems wrong because I was just using the distance, not filtering the candidates by the first letter, which turns out to return fairly dissimilar words.

2.2 Question 10
~~~~~~~~~~~~~~~

For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:

`Jaccard distance <https://en.wikipedia.org/wiki/Jaccard_index>`_ on the 4-grams of the two words.

**This function should return a list of length three:** ``['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']``.

.. code:: ipython

    def answer_ten(entries=['cormulent', 'incendenece', 'validrate']):
        """gets the neares words using jaccard-distance with 4-grams

        Args:
         entries (list): words to find nearest other word for
    
        Returns:
         list: nearest words found
        """
        return jaccard(entries, 4)
    
    print(answer_ten())

::

    ['cormus', 'incendiary', 'valid']

2.3 Question 11
~~~~~~~~~~~~~~~

For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:

`Edit (Levenshtein) distance  <https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance>`_ on the two words with transpositions.

**This function should return a list of length three:** ``['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']``.

.. code:: ipython

    def answer_eleven(entries=['cormulent', 'incendenece', 'validrate']):
        """gets the nearest words based on Levenshtein distance

        Args:
         entries (list[str]): words to find closest words to

        Returns:
         list[str]: nearest words to the entries
        """
        outcomes = []
        for entry in entries:
            distances = ((edit_distance(entry,
                                        word), word)
                         for word in correct_spellings)
            closest = min(distances)
            outcomes.append(closest[1])
        return outcomes
    
    print(answer_eleven())

::

    ['corpulent', 'intendence', 'validate']

