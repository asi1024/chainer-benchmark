import numpy

import chainer.functions as F

from chainer.utils import conv

from benchmarks.functions import FunctionBenchmark
from benchmarks.utils import backends


@backends('gpu', 'gpu-cudnn', 'cpu')
class UnpoolingND(FunctionBenchmark):
    def setup(self):
        xp = self.xp

        # Prepare test data.
        batches = 16
        channels = 4
        in_size = (32, 32, 32)
        ksize = 4

        out_size = tuple([conv.get_deconv_outsize(
            size, ksize, ksize, 0, cover_all=True) for size in in_size])
        x_shape = (batches, channels) + in_size
        gy_shape = (batches, channels) + out_size

        x = xp.arange(numpy.prod(x_shape), dtype=xp.float32).reshape(x_shape)
        xp.random.shuffle(x)
        x = 2 * x / xp.float32(x.size) - 1
        gy = xp.random.uniform(-1, -1, gy_shape).astype(xp.float32)

        # Setup benchmark.
        self.setup_benchmark(F.unpooling_nd, (x, ksize), gy)

    def time_forward(self):
        self.forward()

    def time_backward(self):
        self.backward()
