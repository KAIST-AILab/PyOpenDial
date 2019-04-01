import functools
import logging
import random

from multipledispatch import dispatch

from datastructs.assignment import Assignment

dispatch_namespace = dict()


class InferenceUtils:
    """
    Utility functions for inference operations.
    """
    _eps = 1e-2

    # logger
    log = logging.getLogger('PyOpenDial')

    @staticmethod
    @dispatch(dict, namespace=dispatch_namespace)
    def normalize(distrib):
        """
        Normalise the given probability distribution (assuming no conditional variables).

        :param distrib: the distribution to normalise
        :return: the normalised distribution
        """
        if isinstance(distrib, dict):
            total_prob = sum(distrib.values())
            if total_prob <= 0.:
                # TODO: check bug > is this a sufficient condition for the error cases?
                # TODO: check bug > Is just returning distrib enough?
                InferenceUtils.log.warning("all assignments in the distribution have a zero probability, cannot be normalised")
                return distrib

            for key, value in distrib.items():
                distrib[key] = distrib[key] / total_prob
            return distrib

    @staticmethod
    @dispatch(list, namespace=dispatch_namespace)
    def normalize(init_probs):
        """
        Normalises the double array (ensuring that the sum is equal to 1.0).

        :param init_probs: the unnormalised values
        :return: the normalised values
        """
        total_prob = sum(init_probs)
        if total_prob > 0. + InferenceUtils._eps:
            for idx in range(len(init_probs)):
                init_probs[idx] = init_probs[idx] / total_prob

                # TODO: check refactor > do we have to return distrib with new instance?
        return init_probs

    @staticmethod
    @dispatch(dict, namespace=dispatch_namespace)
    def get_all_combinations(values_matrix):
        """
        Generates all possible assignment combinations from the set of values provided
        as parameters -- each variable being associated with a set of alternative values.
        NB: use with caution, computational complexity is exponential!

        :param values_matrix: the set of values to combine
        :return: the list of all possible combinations
        """
        combinations = set()
        combinations.add(Assignment())

        for label in values_matrix.keys():
            values = values_matrix.get(label)
            partial_combinations = set()
            for assignment in combinations:
                for value in values:
                    new_assignment = Assignment(assignment, label, value)
                    partial_combinations.add(new_assignment)
            combinations = partial_combinations

        return combinations

    @staticmethod
    @dispatch(dict, int, namespace=dispatch_namespace)
    def get_n_best(init_table, n_best):
        """
        Returns a smaller version of the initial table that only retains the N
        elements with a highest value

        :param init_table: the full initial table
        :param n_best: the number of elements to retain
        :return: the resulting subset of the table
        """
        if n_best < 1:
            InferenceUtils.log.warning("nbest should be >=, but is %s" % n_best)
            raise ValueError()

        entries = [entry for entry in init_table.items()]

        # TODO: 버그인지 확인.
        # random.shuffle(entries)

        def n_best_entry_comparator(a, b):
            value = a[1] - b[1]
            if abs(value) < 0.0001:
                return 1 if random.randint(0, 1) == 1 else -1
            else:
                return int(value * 10000000)

        entries = sorted(entries, key=functools.cmp_to_key(n_best_entry_comparator), reverse=True)

        table = dict()
        cnt = 0
        for entry in entries:
            if cnt < n_best:
                table[entry[0]] = entry[1]
                cnt += 1

        return table

    @staticmethod
    @dispatch(dict, Assignment, float, namespace=dispatch_namespace)
    def get_ranking(init_table, assignment, min_diff):
        """
        Returns the ranking of the given assignment in the table, assuming an ordering
        of the table in descending order.

        :param init_table: the table
        :param assignment: the assignment to find
        :param min_diff: the minimum difference between values
        :return: the index in the ordered table, or -1 if the element is not in the table
        """
        entries = [entry for entry in init_table.items()]

        def ranking_entry_comparator(a, b):
            value = a[1] - b[1]
            if abs(value) < min_diff:
                return 0
            else:
                return int(value * 10000000)

        entries = sorted(entries, key=functools.cmp_to_key(ranking_entry_comparator), reverse=True)

        for idx, entry in enumerate(entries):
            if entry[0] == assignment:
                return idx

            for next_idx in range(idx + 1, len(entries)):
                next_entry = entries[next_idx]
                if ranking_entry_comparator(entry, next_entry) != 0:
                    break
                if next_entry[0] == assignment:
                    return idx

        return -1
