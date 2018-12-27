using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using NumSharp.Core;
using Proxem.BlasNet;
using System;
using System.Collections.Generic;
using System.Text;

namespace Proxem.NumNet.Benchmark
{
    [CoreJob]
    //[CsvExporter]
    [GroupBenchmarksBy(BenchmarkLogicalGroupRule.ByCategory)]
    [CategoriesColumn]
    public class BaseFunctionsBenchmarkMkl
    {
        private Array<float> numnet_1;
        private Array<float> numnet_2;

        private Array<float> numnet_flat_1;
        private Array<float> numnet_flat_2;

        //private NumPy np;

        private NDArray numsharp_1;
        private NDArray numsharp_2;

        private NDArray numsharp_flat_1;
        private NDArray numsharp_flat_2;

        [Params(100, 500)]
        public int N;

        [GlobalSetup]
        public void Setup()
        {
            // Launching mkl for NumNet (path might need to be change)
            var path = "C:/data/dlls/mkl";
            StartProvider.LaunchMklRt(1, path);

            numnet_1 = NN.Random.Normal(0, 1, N, N);
            numnet_2 = NN.Random.Normal(0, 1, N, N);
            numnet_flat_1 = NN.Random.Normal(0, 1, N * N);
            numnet_flat_2 = NN.Random.Normal(0, 1, N * N);

            //np = new NumPy();
            numsharp_1 = np.random.normal(0, 1, N, N).reshape(new Shape(N, N)); // need reshaping cause there's a bug in 'np.random.normal'
            numsharp_2 = np.random.normal(0, 1, N, N).reshape(new Shape(N, N));
            numsharp_flat_1 = np.random.normal(0, 1, N * N);
            numsharp_flat_2 = np.random.normal(0, 1, N * N);
        }

        [BenchmarkCategory("Dot"), Benchmark]
        public Array<float> NumNetDot() => NN.Dot(numnet_1, numnet_2);

        [BenchmarkCategory("Dot"), Benchmark]
        public NDArray NumSharpDot() => np.dot(numsharp_1, numsharp_2);

        [BenchmarkCategory("Dot"), Benchmark]
        public Array<float> NumNetDotFlat() => NN.Dot(numnet_flat_1, numnet_flat_2);

        [BenchmarkCategory("Dot"), Benchmark]
        public NDArray NumSharpDotFlat() => np.dot(numsharp_flat_1, numsharp_flat_2);

        [BenchmarkCategory("Maths"), Benchmark]
        public Array<float> NumNetLog() => NN.Log(numnet_1);

        [BenchmarkCategory("Maths"), Benchmark]
        public NDArray NumSharpLog() => np.log(numsharp_1);

        [BenchmarkCategory("Operations"), Benchmark]
        public Array<float> NumNetDiff() => numnet_1 - numnet_2;

        [BenchmarkCategory("Operations"), Benchmark]
        public NDArray NumSharpDiff() => numsharp_1 - numsharp_2;

        [BenchmarkCategory("Operations"), Benchmark]
        public Array<float> NumNetAdd() => numnet_1 + numnet_2;

        [BenchmarkCategory("Operations"), Benchmark]
        public NDArray NumSharpAdd() => numsharp_1 + numsharp_2;

        [BenchmarkCategory("Operations"), Benchmark]
        public Array<float> NumNetHadamard() => numnet_1 * numnet_2;

        [BenchmarkCategory("Operations"), Benchmark]
        public NDArray NumSharpHadamard() => numsharp_1 * numsharp_2;

        [BenchmarkCategory("Operations"), Benchmark]
        [Arguments(1.5f)]
        [Arguments(-2.8f)]
        public Array<float> NumNetScalarMul(float lambda) => numnet_1 * lambda;

        [BenchmarkCategory("Operations"), Benchmark]
        [Arguments(1.5f)]
        [Arguments(-2.8f)]
        public NDArray NumSharpScalarMul(float lambda) => numsharp_1 * lambda;

        [BenchmarkCategory("Base"), Benchmark]
        public Array<float> NumNetArgmax() => NN.Argmax(numnet_1);

        [BenchmarkCategory("Base"), Benchmark]
        public NDArray NumSharpArgmax() => np.amax(numsharp_1);

        [BenchmarkCategory("Base"), Benchmark]
        public void NumNetArgmaxAxis()
        {
            var a = NN.Argmax(numnet_1, 0);
            var b = NN.Argmax(numnet_1, 1);
        }

        [BenchmarkCategory("Base"), Benchmark]
        public void NumSharpArgmaxAxis()
        {
            np.amax(numsharp_1, 0);
            np.amax(numsharp_1, 1);
        }
    }
}
