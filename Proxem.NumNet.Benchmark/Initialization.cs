using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using NumSharp.Core;
using Proxem.BlasNet;
using System;
using System.Collections.Generic;
using System.Text;

namespace Proxem.NumNet.Benchmark
{
    [GroupBenchmarksBy(BenchmarkLogicalGroupRule.ByCategory)]
    [CategoriesColumn]
    [CoreJob]
    [CsvExporter]
    public class InitBenchmark
    {
        private NumPy np;

        [GlobalSetup]
        public void Setup()
        {
            // Launching mkl for NumNet (path might need to be change)
            var path = "C:/data/dlls/mkl";
            StartProvider.LaunchMklRt(1, path);

            // Creating NumPy for NumSharp
            np = new NumPy();
        }

        [BenchmarkCategory("Gaussian"), Benchmark(Baseline = true)]
        [Arguments(100, 200)]
        [Arguments(1000, 2000)]
        public void NumNetInitNormal(int n, int m)
        {
            var a = NN.Random.Normal(-.4f, 1.1f, n, m);
        }

        [BenchmarkCategory("Gaussian"), Benchmark]
        [Arguments(100, 200)]
        [Arguments(1000, 2000)]
        public void NumSharpInitNormal(int n, int m)
        {
            var a = np.random.normal(-.4f, 1.1f, n, m);
        }

        [BenchmarkCategory("Arange"), Benchmark(Baseline = true)]
        [Arguments(130, 130)]
        [Arguments(200, 10)]
        public void NumNetInitArange(int n, int m)
        {
            var a = NN.Range(n * m).Reshape(n, m);
        }

        [BenchmarkCategory("Arange"), Benchmark]
        [Arguments(130, 130)]
        [Arguments(200, 10)]
        public void NumSharpInitArange(int n, int m)
        {
            var a = np.arange(n * m).reshape(n, m);
        }

        [BenchmarkCategory("Constant"), Benchmark]
        [Arguments(1000, 2000)]
        [Arguments(2000, 1000)]
        public void NumNetInitOnes(int n, int m)
        {
            var a = NN.Ones(n, m);
        }

        [BenchmarkCategory("Constant"), Benchmark]
        [Arguments(1000, 2000)]
        [Arguments(2000, 1000)]
        public void NumSharpInitOnes(int n, int m)
        {
            var a = np.ones(n, m);
        }

        [BenchmarkCategory("Constant"), Benchmark]
        [Arguments(1000, 2000)]
        [Arguments(2000, 1000)]
        public void NumNetInitZeros(int n, int m)
        {
            var a = NN.Zeros(n, m);
        }

        [BenchmarkCategory("Constant"), Benchmark]
        [Arguments(1000, 2000)]
        [Arguments(2000, 1000)]
        public void NumSharpInitZeros(int n, int m)
        {
            var a = np.zeros(n, m);
        }
    }
}
