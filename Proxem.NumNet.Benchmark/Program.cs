using System;
using BenchmarkDotNet.Running;

namespace Proxem.NumNet.Benchmark
{
    class Program
    {
        static void Main(string[] args)
        {
            var summary = BenchmarkRunner.Run<BaseFunctionsBenchmarkMkl>();

            Console.ReadLine();
        }
    }
}
