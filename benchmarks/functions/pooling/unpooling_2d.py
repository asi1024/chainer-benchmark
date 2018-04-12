import numpy

import chainer.functions as F

from chainer.utils import conv

from benchmarks.functions import FunctionBenchmark
from benchmarks.utils import backends


@backends('gpu', 'gpu-cudnn', 'cpu')
class Unpooling2D(FunctionBenchmark):
    def setup(self):
        xp = self.xp

        # Prepare test data.
        batches = 16
        channels = 4
        ih, iw = (128, 128)
        ksize = 4

        oh = conv.get_deconv_outsize(ih, ksize, ksize, 0, cover_all=True)
        ow = conv.get_deconv_outsize(iw, ksize, ksize, 0, cover_all=True)
        x_shape = (batches, channels) + (ih, iw)
        gy_shape = (batches, channels) + (oh, ow)

        x = xp.arange(numpy.prod(x_shape), dtype=xp.float32).reshape(x_shape)
        xp.random.shuffle(x)
        x = 2 * x / xp.float32(x.size) - 1
        gy = xp.random.uniform(-1, -1, gy_shape).astype(xp.float32)

        # Setup benchmark.
        self.setup_benchmark(F.unpooling_2d, (x, ksize), gy)

    def time_forward(self):
        self.forward()

    def time_backward(self):
        self.backward()
