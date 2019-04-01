from xml.etree.ElementTree import Element

from multipledispatch import dispatch
import logging
import numpy as np

from scipy.stats import dirichlet

from bn.distribs.density_functions.density_function import DensityFunction


class DirichletDensityFunction(DensityFunction):
    """
    Density function for a Dirichlet distribution. The distribution is defined through
    an array of alpha hyper-parameters.
    """

    # logger
    log = logging.getLogger('PyOpenDial')

    def __init__(self, alpha_list=None):
        if isinstance(alpha_list, np.ndarray):
            """
            Create a new Dirichlet density function with the provided alpha parameters

            :param alpha_list: the hyper-parameters for the density function
            """
            if len(alpha_list) < 2:
                self.log.warning("must have at least 2 alphas")
                raise ValueError()

            for alpha in alpha_list:
                if alpha <= 0:
                    self.log.warning("alphas of the Dirichlet distribution are not well formed")
                    raise ValueError()

            # hyper-parameters
            self._alpha_list = alpha_list
            self._density_func = dirichlet(alpha_list)
        else:
            raise NotImplementedError()

    @dispatch(np.ndarray)
    def get_density(self, x):
        """
        Returns the density for a given point x. The dimensionality of x must
        correspond to the dimensionality of the density function.

        :param x: a given point
        :return: the density for the point
        """
        if len(x) == len(self._alpha_list):
            return self._density_func.pdf(x)

        self.log.warning("incompatible sizes: ", len(x), "!=", len(self._alpha_list))
        return 0.0

    @dispatch()
    def get_dimensions(self):
        """
        Returns the dimensionality of the density function

        :return: the dimensionality
        """
        return len(self._alpha_list)

    @dispatch()
    def sample(self):
        """
        Returns a sampled value for the density function.

        :return: the sampled point.
        """
        return self._density_func.rvs()[0]

    def __copy__(self):
        """
        Copies the density function (keeping the same alpha-values).

        :return: the copied function
        """
        return DirichletDensityFunction(self._alpha_list)

    def __str__(self):
        """
        Returns the name and hyper-parameters of the distribution

        :return: the string for the density function
        """
        return 'Dirichlet(' + str(self._alpha_list) + ')'

    @dispatch(int)
    def discretize(self, nb_buckets):
        """
        Returns a discretised version of the Dirichlet. The discretised table is
        simply a list of X sampled values from the Dirichlet, each value having a
        probability 1/X.

        :param nb_buckets:
        :return: the discretised version of the density function.
        """
        table = dict()
        for idx in range(nb_buckets):
            table[tuple(self.sample())] = 1.0 / nb_buckets

        return table

    @dispatch()
    def get_mean(self):
        """
        Returns the mean of the Dirichlet.

        :return: the mean value.
        """
        return self._density_func.mean()

    @dispatch()
    def get_variance(self):
        """
        Returns the variance of the Dirichlet.

        :return: the variance.
        """
        return self._density_func.var()

    @dispatch(np.ndarray)
    def get_cdf(self, x):
        """
        Throws an exception (calculating the CDF of a Dirichlet is quite hard and not currently implemented).
        """
        raise NotImplementedError("Currently not implemented (CDF of Dirichlet has apparently no closed-form solution)")

    def __hash__(self):
        """
        Returns the hashcode for the distribution.

        :return: the hashcode
        """
        return -32 + hash(self._alpha_list)

    def generate_xml(self):
        distrib_element = Element('distrib')
        distrib_element.set('type', 'dirichlet')

        for alpha in self._alpha_list:
            alpha_element = Element('alpha')
            alpha_element.text = str(alpha)
            distrib_element.append(alpha_element)

        return [distrib_element]
