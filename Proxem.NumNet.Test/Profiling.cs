/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Proxem.BlasNet;
using Proxem.NumNet.Single;

namespace Proxem.NumNet.Test
{
    [TestClass]
    public class Profiling
    {
        [Ignore]
        [TestMethod, TestCategory("Profiling")]
        public void CompareElementWisePerformance()
        {
            Trace.Listeners.Add(new ConsoleTraceListener());

            Func<float, float, float> f = (x, y) => x + y;
            var clock = new Stopwatch();
#if DEBUG
            Trace.WriteLine($"Testing on DEBUG");
#else
            Trace.WriteLine($"Testing on RELEASE");
#endif
            Trace.WriteLine($"Testing on {Blas.NThreads} threads");

            for (int i = 0; i < 300; ++i)
            {
                int n = i + 1;
                var a = NN.Random.Uniform(-1f, 1f, n, n);
                var b = NN.Random.Uniform(-1f, 1f, n, n);
                var r = NN.Zeros(n, n);

                var size = a.Size;
                // estimating loop count for this size
                NN.MIN_SIZE_FOR_PARELLELISM = size * 2;
                var loopCount = 0;
                clock.Restart();

                while (clock.ElapsedMilliseconds < 1000)
                {
                    NN.Apply(a, b, f, result: r);
                    ++loopCount;
                }
                Trace.WriteLine($"doing {loopCount} loops for size {size}");

                // profiling Normal
                clock.Restart();
                for (int _ = 0; _ < loopCount; _++)
                    NN.Apply(a, b, f, result: r);
                var time = clock.ElapsedMilliseconds;
                var avg = (double)time / loopCount;

                // profiling Parrellized
                NN.MIN_SIZE_FOR_PARELLELISM = 0;
                clock.Restart();
                for (int _ = 0; _ < loopCount; _++)
                    NN.Apply(a, b, f, result: r);
                var timePar = clock.ElapsedMilliseconds;
                var avgPar = (double)timePar / loopCount;

                clock.Restart();
                for (int _ = 0; _ < loopCount; _++)
                    a.Add(b, result: r);
                var timeAdd = clock.ElapsedMilliseconds;
                var avgAdd = (double)timeAdd / loopCount;

                clock.Restart();
                for (int _ = 0; _ < loopCount; _++)
                    Blas.vadd(size, a.Values, 0, b.Values, 0, r.Values, 0);
                var timeBlas = clock.ElapsedMilliseconds;
                var avgBlas = (double)timeBlas / loopCount;

                var message = $"On size {size}, avg time: {avg:F4}ms \t with parallelism {avgPar:F4}ms \t with Add {avgAdd:F4}ms \t with Blas {avgBlas:F4}ms.";
                Trace.WriteLine(message);
            }

            throw new Exception("see output for profiler results");
        }
    }
}
