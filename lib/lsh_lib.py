# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from datasketch import MinHash, MinHashLSH


class ClusteringLSHLib(object):
    """
    Suppose you have a very large collection of sets. Giving a query, which is
    also a set, you want to find sets in your collection that have Jaccard
    similarities above certain threshold, and you want to do it with many other
    queries. To do this efficiently, you can create a MinHash for every set,
    and when a query comes, you compute the Jaccard similarities between the
    query MinHash and all the MinHash of your collection, and return the sets
    that satisfy your threshold.

    *** Read more via : https://ekzhu.github.io/datasketch/lsh.html

    """

    def __init__(self, threshold=0.9, num_perm=128):
        """
        Init
        Args:
            threshold (float): The Jaccard similarity threshold between 0.0 and
            1.0. The initialized MinHash LSH will be optimized for the threshold
             by minizing the false positive and false negative.
            num_perm (int): The number of permutation functions used
            by the MinHash to be indexed
        """
        self.threshold = threshold
        self.num_perm = num_perm
        self.lsh_server = MinHashLSH(threshold=threshold, num_perm=num_perm)

    def get_lsh_server(self):
        return self.lsh_server

    def compute_min_hash_lsh(self, terms):
        """
        Compute min hash LSH of a set of tokens

        Args:
            terms (set): set of unique terms

        Returns:
            (MinHash): min hash LSH value

        """
        m = MinHash(num_perm=self.num_perm)
        for e in terms:
            m.update(e.encode('utf8'))
        return m

    def compute_min_hash_lsh_over_data(self, record_ids, data):
        """
        Compute min hash of each document from given record Ids and data
        Args:
            record_ids (list[int]): list of given record Id
            data (list[list[str]]): list of content belonged to record Ids above

        Returns:
            lsh_vals (list[MinHash]): list of min hash value

        """
        # make sure docId is unique over the corpus
        assert len(set(record_ids)) == len(record_ids)

        print(record_ids[0:2], data[0:2])
        print(record_ids[-1], data[-1])

        # for each record compute the hash
        lsh_vals = [
            self.compute_min_hash_lsh(terms=set(terms))
            for terms in data
        ]
        # TODO: convert to parallel
        for record_id, hash_val in zip(record_ids, lsh_vals):
            idx = "{}".format(record_id)
            # update the hash document to whole corpus
            self.lsh_server.insert(idx, hash_val)
        return lsh_vals

    def query_duplicated_record(self, query):
        """
        Query to LSH corpus for getting duplicated record Id
        Args:
            query (MinHash):

        Returns:
            result (list[int]): record Id

        """
        result = self.lsh_server.query(query)
        result = [idx for idx in result]
        return sorted(result)

    def clustering(self, df):
        """
        Query every document in corpus to find duplicated content

        Args:
            lsh_vals (list[MinHash]): list of LSH hash
        Returns:
            duplicated_ids (list[int]): list of duplicated record Ids

        """
        # duplicated_ids = []
        # for idx in range(len(lsh_vals)):
        #     result = self.query_duplicated_document(lsh_vals[idx])
        #     if len(result) > 1:
        #         duplicated_ids.append(result)
        #
        # return duplicated_ids
        print(df.head())
