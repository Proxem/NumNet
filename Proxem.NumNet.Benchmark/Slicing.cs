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
    public class SlicingBenchmark
    {
        private Array<float> _numnet_3d;
        private NDArray _numsharp_3d;

        private Array<float> _numnet_2d;
        private NDArray _numsharp_2d;

        [GlobalSetup]
        public void Setup()
        {
            // Launching mkl for NumNet (path might need to be change)
            var path = "C:/data/dlls/mkl";
            StartProvider.LaunchMklRt(1, path);
            //var np = new NumPy(); deprecated

            _numnet_3d = NN.Random.Normal(0f, 1f, 10, 8, 12);
            _numsharp_3d = np.random.normal(0f, 1f, 10, 8, 12);

            _numnet_2d = NN.Random.Normal(-1f, 0.4f, 50, 20);
            _numsharp_2d = np.random.normal(-1f, 0.4f, 50, 20);
        }

        [BenchmarkCategory("Slicing"), Benchmark]
        public void SlicingNumNet()
        {
            var slice_1 = _numnet_3d[Slicer._, 0, 3];
            var slice_2 = _numnet_3d[1, Slicer._, 2];
            var slice_3 = _numnet_3d[2, 4, Slicer._];
        }

        //[BenchmarkCategory("Slicing"), Benchmark]
        //public void SlicingNumSharp()
        //{
        //    var slice_1 = _numsharp_3d[new Shape(1)];
        //}
    }
}
