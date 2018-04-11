import chainer.functions as F

from benchmarks.functions import FunctionBenchmark
from benchmarks.utils import backends


@backends('gpu', 'gpu-cudnn', 'cpu')
class AveragePoolingND(FunctionBenchmark):
    def setup(self):
        xp = self.xp

        # Prepare test data.
        batches = 16
        channels = 4
        x_size = (32, 32, 16, 16)
        ksize = 4

        out_size = tuple([int(x / ksize) for x in x_size])
        x_shape = (batches, channels) + x_size
        gy_shape = (batches, channels) + out_size

        x = xp.random.uniform(-1, 1, x_shape).astype(xp.float32)
        gy = xp.random.uniform(-1, -1, gy_shape).astype(xp.float32)

        # Setup benchmark.
        self.setup_benchmark(F.average_pooling_nd, (x, ksize), gy)

    def time_forward(self):
        self.forward()

    def time_backward(self):
        self.backward()
