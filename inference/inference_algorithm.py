import abc
from collections import Collection
import logging

from multipledispatch import dispatch

from bn.b_network import BNetwork
from datastructs.assignment import Assignment
from inference.query import ProbQuery, UtilQuery, ReduceQuery


class InferenceAlgorithm:
    """
     Generic interface for probabilistic inference algorithms. Three distinct types of
     queries are possible:

     - probability queries of the form P(X1,...,Xn)
     - utility queries of the form U(X1,...,Xn)
     - reduction queries where a Bayesian network is reduced to a subset of variables
      X1,...,Xn

     The interface contains 3 abstract methods (queryProb, queryUtil and reduce) that
     must be specified by all implementing classes. In addition, a set of default
     methods provide alternative ways to call the inference process (e.g. with one or
     several query variables, with or without evidence, etc.).
    """
    __metaclass__ = abc.ABCMeta

    log = logging.getLogger('PyOpenDial')

    @dispatch(ProbQuery)
    @abc.abstractmethod
    def query_prob(self, query):
        """
        Computes the probability distribution for the query variables given the
        provided evidence, all specified in the query.

        :param query: the full query
        :return: the resulting probability distribution failed to deliver a result
        """
        raise NotImplementedError()

    @dispatch(BNetwork, Collection, Assignment)
    def query_prob(self, network, query_vars, evidence):
        """
        Computes the probability distribution for the query variables given the
        provided evidence.

        :param network: the Bayesian network on which to perform the inference
        :param query_vars: the collection of query variables
        :param evidence: the evidence
        :return: the resulting probability distribution failed to deliver a result
        """
        return self.query_prob(ProbQuery(network, query_vars, evidence))

    @dispatch(BNetwork, Collection)
    def query_prob(self, network, query_vars):
        """
        Computes the probability distribution for the query variables, assuming no
        additional evidence.

        :param network: the Bayesian network on which to perform the inference
        :param query_vars: the collection of query variables
        :return: the resulting probability distribution failed to deliver a result
        """
        return self.query_prob(ProbQuery(network, query_vars, Assignment()))

    @dispatch(BNetwork, str, Assignment)
    def query_prob(self, network, query_var, evidence):
        """
        Computes the probability distribution for the query variable given the
        provided evidence.

        :param network: the Bayesian network on which to perform the inference
        :param query_var: the (unique) query variable
        :param evidence: the evidence
        :return: the resulting probability distribution for the variable inference
        process failed to deliver a result
        """
        return self.query_prob(ProbQuery(network, [query_var], evidence)).get_marginal(query_var)

    @dispatch(BNetwork, str)
    def query_prob(self, network, query_var):
        """
        Computes the probability distribution for the query variable given the
        provided evidence.

        :param network: the Bayesian network on which to perform the inference
        :param query_var: the (unique) query variable
        :return: the resulting probability distribution for the variable inference
        process failed to deliver a result
        """
        return self.query_prob(network, query_var, Assignment())

    @dispatch(UtilQuery)
    @abc.abstractmethod
    def query_util(self, query):
        """
        Computes the utility table for the query variables (typically action
        variables), given the provided evidence.

        :param query: the full query
        return the resulting utility table deliver a result
        """
        raise NotImplementedError()

    @dispatch(BNetwork, Collection, Assignment)
    def query_util(self, network, query_vars, evidence):
        """
        Computes the utility table for the query variables (typically action
        variables), given the provided evidence.

        :param network: the Bayesian network on which to perform the inference
        :param query_vars: the query variables (usually action variables)
        :param evidence: the additional evidence
        :return: the resulting utility table for the query variables process failed to
               deliver a result
        """
        return self.query_util(UtilQuery(network, query_vars, evidence))

    @dispatch(BNetwork, Collection)
    def query_util(self, network, query_vars):
        """
        Computes the utility table for the query variables (typically action
        variables), assuming no additional evidence.

        :param network: the Bayesian network on which to perform the inference
        :param query_vars: the query variables (usually action variables)
        :return: the resulting utility table for the query variables process failed to
               deliver a result
        """
        return self.query_util(UtilQuery(network, query_vars, Assignment()))

    @dispatch(BNetwork, str, Assignment)
    def query_util(self, network, query_var, evidence):
        """
        Computes the utility table for the query variable (typically an action
        variable), given the provided evidence

        param network: the Bayesian network on which to perform the inference
        param query_var: the query variable
        param evidence: the additional evidence
        return the resulting utility table for the query variable process failed to
               deliver a result
        """
        return self.query_util(UtilQuery(network, [query_var], evidence))

    @dispatch(BNetwork, str)
    def query_util(self, network, query_var):
        """
        Computes the utility table for the query variable (typically an action
        variable), assuming no additional evidence.

        :param network: the Bayesian network on which to perform the inference
        :param query_var: the query variable
        return the resulting utility table for the query variable process failed to
               deliver a result
        """
        return self.query_util(UtilQuery(network, [query_var], Assignment()))

    @dispatch(ReduceQuery)
    def reduce(self, query):
        """
        Generates a new Bayesian network that only contains a subset of variables in
        the original network and integrates the provided evidence.

        :param query: the full reduction query
        :return: the reduced Bayesian network deliver a result
        """
        raise NotImplementedError()

    @dispatch(BNetwork, Collection, Assignment)
    def reduce(self, network, query_vars, evidence):
        """
        Generates a new Bayesian network that only contains a subset of variables in
        the original network and integrates the provided evidence.

        :param network: the original Bayesian network
        :param query_vars: the variables to retain
        :param evidence: the additional evidence
        :return: the new, reduced Bayesian network deliver a result
        """
        return self.reduce(ReduceQuery(network, query_vars, evidence))

    @dispatch(BNetwork, Collection)
    def reduce(self, network, query_vars):
        """
        Generates a new Bayesian network that only contains a subset of variables in
        the original network, assuming no additional evidence.

        :param network: the original Bayesian network
        :param query_vars: the variables to retain
        :return: the new, reduced Bayesian network deliver a result
        """
        return self.reduce(ReduceQuery(network, query_vars, Assignment()))
