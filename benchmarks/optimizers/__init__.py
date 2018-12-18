import six

import chainer
import cupy

from benchmarks import BenchmarkBase


class Link(chainer.Link):
    def __init__(self, param):
        super(Link, self).__init__()
        with self.init_scope():
            self.p = chainer.Parameter(param)

    def __call__(self, x):
        return x * self.p


class OptimizerBenchmark(BenchmarkBase):

    """The base class for benchmark of optimizers."""

    # Call `test_*` methods only once as `backward()` has a side-effect.
    number = 1

    # Repeat the test for 10 times instead of 3 (`timeit.default_repeat`).
    repeat = 10

    def setup_benchmark(self, optimizer, batch_size, unit_num, dtype):
        """Performs setup of benchmark for optimizers.

        Call this in `setup` method of your benchmark class.
        Note that this function performs forward computation.
        """

        xp = self.xp
        self.optimizer = optimizer

        x = xp.random.uniform(-1, 1, (batch_size, unit_num)).astype(dtype)
        param = xp.random.uniform(-1, 1, unit_num).astype(dtype)
        model = Link(param)
        if xp is cupy:
            model.to_gpu()

        x = chainer.Variable(x)
        y = model(x)
        y.zerograd()
        y.backward()
        optimizer.setup(model)

    def update(self, n_times):
        """Runs optimizer.update()."""

        optimizer = self.optimizer

        for i in six.moves.range(n_times):
            optimizer.update()
