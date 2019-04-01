import logging
import math
import multiprocessing
import threading
import traceback
from contextlib import closing
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool

from multipledispatch import dispatch

from bn.distribs.continuous_distribution import ContinuousDistribution
from bn.nodes.action_node import ActionNode
from bn.nodes.chance_node import ChanceNode
from bn.nodes.utility_node import UtilityNode
from inference.approximate.intervals import Intervals
from inference.approximate.sample import Sample
from inference.query import Query
from utils.py_utils import current_time_millis


class LikelihoodWeighting:
    """
    Sampling process (based on likelihood weighting) for a particular query.
    """
    log = logging.getLogger('PyOpenDial')

    _weight_threshold = 0.0001

    def __init__(self, query, nr_samples, max_sampling_time):
        if not isinstance(query, Query) or not isinstance(nr_samples, int) or not isinstance(max_sampling_time, int):
            raise NotImplementedError("UNDEFINED PARAMETERS")
        """
        Creates a new sampling query with the given arguments and starts sampling
        (using parallel streams).

        :param query: the query to answer
        :param nr_samples: the number of samples to collect
        :param max_sampling_time: maximum sampling time (in milliseconds)
        """
        self._query = query
        self._evidence = query.get_evidence()
        self._query_vars = query.get_query_vars()

        self._sorted_nodes = query.get_filtered_sorted_nodes()
        self._sorted_nodes.sort(reverse=True)

        self._sample_cnt = 0
        self._max_sampling_time = max_sampling_time

        self._samples = []
        for _ in range(nr_samples):
            self._samples.append(self.sample())

        # nr_processes = multiprocessing.cpu_count() - 2
        # if nr_processes < 1:
        #     nr_processes = 1
        #
        # if nr_processes == 1:
        #     nr_samples_per_process = [nr_samples]
        # else:
        #     nr_samples_per_process = [nr_samples // nr_processes] * (nr_processes) # - 1) + [nr_samples % nr_processes]
        #
        # threads = []
        # self._thread_results = [None] * nr_processes
        #
        # for thread_idx in range(nr_processes):
        #     thread = threading.Thread(target=self._sampling_process, args=(thread_idx, nr_samples_per_process[thread_idx]))
        #     threads.append(thread)
        #
        # for thread in threads:
        #     thread.start()
        #
        # for thread in threads:
        #     thread.join()
        #
        # self._samples = list()
        # for samples in self._thread_results:
        #     self._samples.extend(samples)

    #     with closing(Pool(processes=nr_processes)) as pool:
    #         if nr_processes == 1:
    #             nr_samples_per_process = [nr_samples]
    #         else:
    #             nr_samples_per_process = [nr_samples // nr_processes] * (nr_processes - 1) + [nr_samples % nr_processes]
    #
    #         samples_from_processes = pool.map(self._sampling_process, nr_samples_per_process)
    #         self._samples = [sample for samples_from_process in samples_from_processes for sample in samples_from_process]
    #         pool.terminate()

    def _sampling_process(self, thread_idx, nr_samples):
        """
        Sub-procedure for sample on different processes.

        :param nr_samples: number of samples to sample
        :return: samples acquired
        """
        print('thread idx: ', thread_idx)
        result = []
        start_time = current_time_millis()
        while len(result) < nr_samples:
            sample = self.sample()
            if sample.get_weight() > LikelihoodWeighting._weight_threshold:
                result.append(sample)
            elapsed_time = current_time_millis() - start_time
            if elapsed_time > self._max_sampling_time:
                break

        self._thread_results[thread_idx] = result

    def __str__(self):
        return '%s (%d samples already collected)' % (str(self._query), len(self._samples))

    def get_samples(self):
        """
        Returns the collected samples

        :return: the collected samples
        """
        self._redraw_samples()
        return self._samples

    def sample(self):
        """
        Runs the sample collection procedure until termination (either due to a
        time-out or the collection of a number of samples = nbSamples). The method
        loops until terminate() is called, or enough samples have been collected.

        :return: the resulting sample
        """
        sample = Sample()
        try:
            for node in self._sorted_nodes:
                node_id = node.get_id()

                if len(node.get_input_node_ids()) == 0 and self._evidence.contains_var(node_id):
                    sample.add_pair(node_id, self._evidence.get_value(node_id))
                elif isinstance(node, ChanceNode):
                    self.sample_chance_node(node, sample)
                elif isinstance(node, ActionNode):
                    self.sample_action_node(node, sample)
                elif isinstance(node, UtilityNode):
                    utility = node.get_utility(sample)
                    sample.add_utility(utility)

            sample.trim(self._query_vars)
        except Exception as e:
            self.log.warning('exception caught: ' + str(e))
            traceback.print_tb(e.__traceback__)

        return sample

    @dispatch(ChanceNode, Sample)
    def sample_chance_node(self, node, sample):
        """
        Samples the given chance node and add it to the sample. If the variable is
        part of the evidence, updates the weight.

        :param node: the chance node to sample
        :param sample: sample
        """
        node_id = node.get_id()

        if not self._evidence.contains_var(node_id):
            value = node.sample(sample)
            sample.add_pair(node_id, value)
        else:
            evidence_value = self._evidence.get_value(node_id)
            distrib = node.get_distrib()

            if isinstance(distrib, ContinuousDistribution):
                evidence_prob = distrib.get_prob_density(evidence_value)
            else:
                evidence_prob = node.get_prob(sample, evidence_value)

            try:
                log_prob = math.log(evidence_prob)
            except:
                log_prob = -math.inf

            sample.add_log_weight(log_prob)
            sample.add_pair(node_id, evidence_value)

    @dispatch(ActionNode, Sample)
    def sample_action_node(self, node, sample):
        """
        * Samples the action node. If the node is part of the evidence, simply add it to
        * the sample. Else, samples an action at random.
        *
        :param node: the action node
        :param sample: the weighted sample to extend
        """
        node_id = node.get_id()
        if not self._evidence.contains_var(node_id) and len(node.get_input_node_ids()) == 0:
            value = node.sample()
            sample.add_pair(node_id, value)
        else:
            value = self._evidence.get_value(node_id)
            sample.add_pair(node_id, value)

    def _redraw_samples(self):
        """
        Redraw the samples according to their weight. The number of redrawn samples is
        the same as the one given as argument.
        """
        try:
            intervals = Intervals(self._samples, lambda x: x.get_weight())

            new_samples = list()
            for _ in range(len(self._samples)):
                new_samples.append(intervals.sample())

            self._samples = new_samples
        except Exception as e:
            self.log.warning('could not redraw samples: ' + str(e))
            traceback.print_tb(e.__traceback__)
