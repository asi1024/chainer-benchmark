import numpy

import chainer.functions as F

from benchmarks.functions import FunctionBenchmark
from benchmarks.utils import backends


@backends('gpu', 'gpu-cudnn', 'cpu')
class SpatialPyramidPooling2D(FunctionBenchmark):
    def setup(self):
        xp = self.xp

        # Prepare test data.
        batches = 16
        channels = 4
        x_size = (128, 128)
        pyramid_height = 4
        output_dim = 340

        x_shape = (batches, channels) + x_size
        gy_shape = (batches, output_dim, 1, 1)

        x = xp.arange(numpy.prod(x_shape), dtype=xp.float32).reshape(x_shape)
        xp.random.shuffle(x)
        x += xp.random.uniform(0.4, 0.6, x_shape).astype(xp.float32)
        x /= numpy.prod(x_shape)
        gy = xp.random.uniform(-1, -1, gy_shape).astype(xp.float32)

        # Setup benchmark.
        self.setup_benchmark(F.spatial_pyramid_pooling_2d,
                             (x, pyramid_height, None, 'max'), gy)

    def time_forward(self):
        self.forward()

    def time_backward(self):
        self.backward()
