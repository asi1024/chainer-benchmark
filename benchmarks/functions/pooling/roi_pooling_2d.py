import numpy

import chainer.functions as F

from benchmarks.functions import FunctionBenchmark
from benchmarks.utils import backends
from benchmarks.utils import parameterize


@backends('gpu', 'cpu')
@parameterize([('batches', [16])])
class ROIPooling2D(FunctionBenchmark):
    def setup(self, batches):
        xp = self.xp

        # Prepare test data.
        channels = 4
        x_size = (2048, 2048)
        out_size = (128, 128)
        rois = xp.array([
            [0, 1, 1, 6, 6],
            [2, 6, 2, 7, 11],
            [1, 3, 1, 5, 10],
            [0, 3, 3, 3, 3]
        ], dtype=xp.float32)
        spatial_scale = 0.6

        x_shape = (batches, channels) + x_size
        gy_shape = (rois.shape[0], channels) + out_size

        x = xp.arange(numpy.prod(x_shape), dtype=xp.float32).reshape(x_shape)
        xp.random.shuffle(x)
        x = 2 * x / xp.float32(x.size) - 1
        gy = xp.random.uniform(-1, -1, gy_shape).astype(xp.float32)

        # Setup benchmark.
        self.setup_benchmark(F.roi_pooling_2d,
                             (x, rois) + out_size + (spatial_scale,), gy)

    def time_forward(self, batches):
        self.forward()

    def time_backward(self, batches):
        self.backward()
