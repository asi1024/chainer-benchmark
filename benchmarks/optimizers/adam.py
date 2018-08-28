import numpy

from chainer import optimizers

from benchmarks.optimizers import OptimizerBenchmark
from benchmarks.utils import backends
from benchmarks.utils import parameterize


@backends('gpu', 'cpu')
@parameterize(
    [('dtype', [numpy.float32, numpy.float64]),
     ('amsgrad', [True, False])])
class Adam(OptimizerBenchmark):
    def setup(self, dtype, amsgrad):
        unit_num = 100000
        batch_size = 32
        optimizer = optimizers.Adam(0.05, amsgrad=amsgrad)

        self.setup_benchmark(optimizer, batch_size, unit_num, dtype)

    def time_update(self, dtype, amsgrad):
        self.update(1000)
