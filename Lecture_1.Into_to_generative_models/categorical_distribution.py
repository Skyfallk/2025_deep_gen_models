import numpy as np


class CategoricalDistribution:
    """Класс реализующий функцию вероятностей pmf для CategoricalDistribution
    [ссылка](https://en.wikipedia.org/wiki/Categorical_distribution)

    Parameters
    ----------
    values : np.array
        массив значений которые может принимать случайная величина
    pd : np.array
        вектор вероятностей размерности len(values) определяющий вероятности
        наступления каждого из элементарных исходов

    Attributes
    ----------
    mapping_dict : dict
        Словарь который выполняет mapping значения элементарного исхода
        на соответствующую ему вероятность появления.

    Examples
    --------
    >>> import numpy as np
    >>> dist = CategoricalDistribution([1, 2, 3], [0.1, 0.3, 0.6])
    >>> dist.pmf(np.array([1, 2, 3]))
    array([0.1, 0.3, 0.6])
    """

    def __init__(self, values, pd):
        self.mapping_dict = dict(zip(values, pd))
        self._vectorized_mapping_fcn = np.vectorize(self._mapping_fcn)

    def _mapping_fcn(self, x):
        """Функция выполняющая mapping значения элементарного исхода
        на соответствующую ему вероятность появления.

        Parameters
        ----------
        x : int
            число соответствующее одному из возможных исходов из self.values

        Returns
        -------
        float
            соответствующее значение вероятности P(x), определяемой из
            self.mapping_dict
        """
        return self.mapping_dict[x]

    def pmf(self, arr):
        """Вычисляет функцию вероятности для входного массива значений
        случайной величины arr.
        Функция использует NumPy's vectorization для эффективного вычисления.

        Parameters
        ----------
        arr : np.array
            входной массива значений случайной величины

        Returns
        -------
        np.array
            соответствующее значения вероятности P(arr), определяемой из
            self.mapping_dict
        """
        return self._vectorized_mapping_fcn(arr)
