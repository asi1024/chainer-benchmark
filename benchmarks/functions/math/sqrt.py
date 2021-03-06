import chainer.functions as F

from benchmarks.functions import UnaryMathFunctionBenchmark
from benchmarks.utils import backends


@backends('gpu', 'cpu')
class SqrtFunc(UnaryMathFunctionBenchmark):

    def setup(self):
        self.setup_benchmark(F.sqrt)

    def time_forward(self):
        self.forward()

    def time_backward(self):
        self.backward()


@backends('gpu', 'cpu')
class RsqrtFunc(UnaryMathFunctionBenchmark):

    def setup(self):
        self.setup_benchmark(F.rsqrt)

    def time_forward(self):
        self.forward()

    def time_backward(self):
        self.backward()
