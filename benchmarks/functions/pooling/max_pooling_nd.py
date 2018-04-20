import chainer.functions as F

from benchmarks.functions import FunctionBenchmark
from benchmarks.utils import backends
from benchmarks.utils import parameterize


@backends('gpu', 'gpu-cudnn', 'cpu')
@parameterize([('batches', [1, 16])])
class MaxPoolingND(FunctionBenchmark):
    def setup(self, batches):
        xp = self.xp

        # Prepare test data.
        channels = 4
        x_size = (32, 32, 16, 16)
        ksize = 4

        out_size = tuple([int(x / ksize) for x in x_size])
        x_shape = (batches, channels) + x_size
        gy_shape = (batches, channels) + out_size

        x = xp.random.uniform(-1, 1, x_shape).astype(xp.float32)
        gy = xp.random.uniform(-1, -1, gy_shape).astype(xp.float32)

        # Setup benchmark.
        self.setup_benchmark(F.max_pooling_nd, (x, ksize), gy)

    def time_forward(self, batches):
        self.forward()

    def time_backward(self, batches):
        self.backward()
