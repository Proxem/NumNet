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
using Microsoft.VisualStudio.TestTools.UnitTesting;

using Proxem.NumNet;
using Proxem.NumNet.Single;

namespace Proxem.NumNet.Test
{
    /// <summary>
    /// http://docs.scipy.org/doc/numpy/reference/generated/numpy.convolve.html
    /// </summary>
    [TestClass]
    public class TestConvolution
    {
        /// <summary>
        /// >>> np.convolve([1, 2, 3], [0, 1, 0.5])
        /// array([ 0. ,  1. ,  2.5,  4. ,  1.5])
        /// </summary>
        [TestMethod]
        public void ConvolveFullAgreesWithNumpy()
        {
            AssertArray.AreAlmostEqual(
                NN.Array(0f, 1f, 2.5f, 4f, 1.5f),
                NN.Array(1f, 2f, 3f).Convolve(NN.Array(0f, 1f, 0.5f), mode: ConvMode.Full)
            );

            AssertArray.AreAlmostEqual(
                NN.Array(0f, 1f, 2.5f, 6f, 6.5f, 8f, 3f),
                NN.Array(0f, 1f, 0.5f, 2f, 1f).Convolve(NN.Array(1f, 2f, 3f), mode: ConvMode.Full)
            );

            AssertArray.AreAlmostEqual(
                NN.Array(0, 1, 2.5f, 3.1f, 4.45f, 5.4f, 4.3f, 12.5f, 4f, 4.5f, 2f),
                NN.Array(1, 2, 0.1f, 0.4f, 5, 1, 2, 1).Convolve(NN.Array(0, 1, 0.5f, 2), mode: ConvMode.Full)
            );
        }

        /// <summary>
        /// Only return the middle values of the convolution.Contains boundary effects, where zeros are taken into account:
        /// >>> np.convolve([1,2,3], [0,1,0.5], 'same')
        /// array([ 1. ,  2.5,  4. ])
        /// </summary>
        [TestMethod]
        public void ConvolveSameAgreesWithNumpy()
        {
            AssertArray.AreAlmostEqual(
                NN.Array(1f, 2.5f, 4f),
                NN.Array(1f, 2f, 3f).Convolve(NN.Array(0f, 1f, 0.5f), mode: ConvMode.Same)
            );

            AssertArray.AreAlmostEqual(
                NN.Array(1f, 2.5f, 6f, 6.5f, 8f),
                NN.Array(0f, 1f, 0.5f, 2f, 1f).Convolve(NN.Array(1f, 2f, 3f), mode: ConvMode.Same)
            );

            AssertArray.AreAlmostEqual(
                NN.Array(1, 2.5f, 3.1f, 4.45f, 5.4f, 4.3f, 12.5f, 4f),
                NN.Array(1, 2, 0.1f, 0.4f, 5, 1, 2, 1).Convolve(NN.Array(0, 1, 0.5f, 2), mode: ConvMode.Same)
            );
        }

        /// <summary>
        /// The two arrays are of the same length, so there is only one position where they completely overlap:
        /// >>> np.convolve([1,2,3],[0,1,0.5], 'valid')
        /// array([ 2.5])
        /// </summary>
        [TestMethod]
        public void ConvolveValidAgreesWithNumpy()
        {
            AssertArray.AreAlmostEqual(
                NN.Array(2.5f),
                NN.Array(1f, 2f, 3f).Convolve(NN.Array(0f, 1f, 0.5f), mode: ConvMode.Valid)
            );

            AssertArray.AreAlmostEqual(
                NN.Array(2.5f, 6f, 6.5f),
                NN.Array(0f, 1f, 0.5f, 2f, 1f).Convolve(NN.Array(1f, 2f, 3f), mode: ConvMode.Valid)
            );

            AssertArray.AreAlmostEqual(
                NN.Array(3.1f, 4.45f, 5.4f, 4.3f, 12.5f),
                NN.Array(1, 2, 0.1f, 0.4f, 5, 1, 2, 1).Convolve(NN.Array(0, 1, 0.5f, 2), mode: ConvMode.Valid)
            );
        }

        [TestMethod]
        public void Convolve2DAgreesWithNumpy()
        {
            var a = NN.Array(new float[,] {
                { 0, 1, 2, 3 },
                { 1, 2, 3, 4 },
                { 2, 3, 4, 5 }
            });
            var k = NN.Array(new float[,] {
                { 0, 1 },
                { 1, 2 }
            });

            var rFull = NN.Array(new float[,] {
                { 0, 0,  1,  2,  3 },
                { 0, 2,  6, 10, 10 },
                { 1, 6, 10, 14, 13 },
                { 2, 7, 10, 13, 10 }
            });
            AssertArray.AreAlmostEqual(rFull, a.Convolve2d(k, mode: ConvMode.Full));

            var rSame = NN.Array(new float[,] {
                { 0, 0,  1,  2 },
                { 0, 2,  6, 10 },
                { 1, 6, 10, 14 }
            });
            AssertArray.AreAlmostEqual(rSame, a.Convolve2d(k, mode: ConvMode.Same));

            var rValid = NN.Array(new float[,] {
                { 2, 6, 10},
                { 6, 10, 14}
            });
            AssertArray.AreAlmostEqual(rValid, a.Convolve2d(k, mode: ConvMode.Valid));
        }
    }

    /// <summary>
    /// http://docs.scipy.org/doc/numpy/reference/generated/numpy.correlate.html
    /// </summary>
    [TestClass]
    public class TestCorrelation
    {
        /// <summary>
        /// >>> np.correlate([1, 2, 3], [0, 1, 0.5])
        /// array([ 3.5])
        /// </summary>
        [TestMethod]
        public void CorrelateValidAgreesWithNumpy()
        {
            AssertArray.AreAlmostEqual(
                NN.Array(3.5f),
                NN.Array(1f, 2f, 3f).Correlate(NN.Array(0f, 1f, 0.5f), mode: ConvMode.Valid)
            );

            AssertArray.AreAlmostEqual(
                NN.Array(3.5f, 8f, 7.5f),
                NN.Array(0f, 1f, 0.5f, 2f, 1f).Correlate(NN.Array(1f, 2f, 3f), mode: ConvMode.Valid)
            );

            AssertArray.AreAlmostEqual(
                NN.Array(7.5f, 8f, 3.5f),
                NN.Array(1f, 2f, 3f).Correlate(NN.Array(0f, 1f, 0.5f, 2f, 1f), mode: ConvMode.Valid)
            );

            AssertArray.AreAlmostEqual(
                NN.Array(7.85f, 11.3f, 6.9f, 10.5f),
                NN.Array(1, 2, 0.1f, 0.4f, 5, 1, 2, 1).Correlate(NN.Array(0, 1, 0.5f, 2, 1), mode: ConvMode.Valid)
            );
        }

        /// <summary>
        /// >>> np.correlate([1, 2, 3], [0, 1, 0.5], "same")
        /// array([ 2. ,  3.5,  3. ])
        /// </summary>
        [TestMethod]
        public void CorrelateSameAgreesWithNumpy()
        {
            AssertArray.AreAlmostEqual(
                NN.Array(2f, 3.5f, 3f),
                NN.Array(1f, 2f, 3f).Correlate(NN.Array(0f, 1f, 0.5f), mode: ConvMode.Same)
            );

            AssertArray.AreAlmostEqual(
                NN.Array(3f, 3.5f, 8f, 7.5f, 4f),
                NN.Array(0f, 1f, 0.5f, 2f, 1f).Correlate(NN.Array(1f, 2f, 3f), mode: ConvMode.Same)
            );

            AssertArray.AreAlmostEqual(
                NN.Array(4f, 7.5f, 8f, 3.5f, 3f),
                NN.Array(1f, 2f, 3f).Correlate(NN.Array(0f, 1f, 0.5f, 2f, 1f), mode: ConvMode.Same)
            );

            AssertArray.AreAlmostEqual(
                NN.Array(4.6f, 2.6f, 7.85f, 11.3f, 6.9f, 10.5f, 4f, 2.5f),
                NN.Array(1, 2, 0.1f, 0.4f, 5, 1, 2, 1).Correlate(NN.Array(0, 1, 0.5f, 2, 1), mode: ConvMode.Same)
            );
        }

        /// <summary>
        /// >>> np.correlate([1, 2, 3], [0, 1, 0.5], "full")
        /// array([ 0.5,  2. ,  3.5,  3. ,  0. ])
        /// </summary>
        [TestMethod]
        public void CorrelateFullAgreesWithNumpy()
        {
            AssertArray.AreAlmostEqual(
                NN.Array(0.5f, 2f, 3.5f, 3f, 0f),
                NN.Array(1f, 2f, 3f).Correlate(NN.Array(0f, 1f, 0.5f), mode: ConvMode.Full)
            );

            AssertArray.AreAlmostEqual(
                NN.Array(0f, 3f, 3.5f, 8f, 7.5f, 4f, 1f),
                NN.Array(0f, 1f, 0.5f, 2f, 1f).Correlate(NN.Array(1f, 2f, 3f), mode: ConvMode.Full)
            );

            AssertArray.AreAlmostEqual(
                NN.Array(1f, 4f, 7.5f, 8f, 3.5f, 3f, 0f),
                NN.Array(1f, 2f, 3f).Correlate(NN.Array(0f, 1f, 0.5f, 2f, 1f), mode: ConvMode.Full)
            );

            AssertArray.AreAlmostEqual(
                NN.Array(1f, 4f, 4.6f, 2.6f, 7.85f, 11.3f, 6.9f, 10.5f, 4f, 2.5f, 1f, 0f),
                NN.Array(1, 2, 0.1f, 0.4f, 5, 1, 2, 1).Correlate(NN.Array(0, 1, 0.5f, 2, 1), mode: ConvMode.Full)
            );
        }

        [TestMethod]
        public void Correlate2dAgreesWithNumpy()
        {
            var a = NN.Array(new float[,] {
                { 0, 1, 2, 3 },
                { 1, 2, 3, 4 },
                { 2, 3, 4, 5 }
            });
            var k = NN.Array(new float[,] {
                { 0, 1 },
                { 1, 2 }
            });

            var rFull = NN.Array(new float[,] {
                 { 0, 2, 5, 8, 3},
                 { 2, 6, 10, 14, 4},
                 { 5, 10, 14, 18, 5},
                 {2, 3, 4, 5, 0}
            });
            AssertArray.AreAlmostEqual(rFull, a.Correlate2d(k, mode: ConvMode.Full));

            var rSame = NN.Array(new float[,] {
                 {6, 10, 14, 4},
                 {10, 14, 18, 5},
                 {3, 4, 5, 0}
            });
            AssertArray.AreAlmostEqual(rSame, a.Correlate2d(k, mode: ConvMode.Same));

            var rValid = NN.Array(new float[,] {
                {6, 10, 14},
                {10, 14, 18}
            });
            AssertArray.AreAlmostEqual(rValid, a.Correlate2d(k, mode: ConvMode.Valid));
        }
    }

    [TestClass]
    public class PoolingTest {

        [TestMethod]
        public void TestPooling()
        {
            var I = new float[,] {
                { 40, 410, 42, 43, 44 },
                { 45, 460, 47, 48, 49 },
                { 50, 51, 520, 53, 54 },
                { 55, 56, 57, 580, 590 }
            };

            var O1 = new float[,]{
                { 460,  48 },
                {  56, 580 }
            };
            AssertArray.AreAlmostEqual(O1, NN.DownSample_MaxPooling2d(I, 2, 2));

            var O2 = new float[,]{
                {460, 49},
                {520, 590}
            };
            AssertArray.AreAlmostEqual(O2, NN.DownSample_MaxPooling2d(I, 2, 3, ignoreBorder: false));

            var O3 = new float[,] {
                { 0, 0, 0, 0, 0 },
                { 0, 460, 0, 48, 0 },
                { 0, 0, 0, 0, 0 },
                { 0, 56, 0, 580, 0 } };
            AssertArray.AreAlmostEqual(O3, NN.Unpooling(O1, I, 2, 2));

            var O4 = new int[,,]{
                {{1,1},{1,3}},
                {{3,1},{3,3}}
            };
            AssertArray.AreEqual(O4, NN.DownSample_MaxPooling2d_IndexArray(I, 2, 2));
        }

        [TestMethod]
        public void ConvolveCustomAgreesWithConvolve()
        {
            int n = 10;
            var x = NN.Random.Uniform(-1f, 1f, n);

            int n_k = 3;
            var k = NN.Random.Uniform(-1f, 1f, n_k);

            var x_k = x.Convolve(k, mode: ConvMode.Full);
            var x_k1 = x.ConvolveCustom(k, NN.Zeros(x_k.Shape), (a, b) => a * b, mode: ConvMode.Full);
            AssertArray.AreAlmostEqual(x_k, x_k1);

            x_k = x.Convolve(k, mode: ConvMode.Same);
            x_k1 = x.ConvolveCustom(k, NN.Zeros(x_k.Shape), (a, b) => a * b, mode: ConvMode.Same);
            AssertArray.AreAlmostEqual(x_k, x_k1);

            x_k = x.Convolve(k, mode: ConvMode.Valid);
            x_k1 = x.ConvolveCustom(k, NN.Zeros(x_k.Shape), (a, b) => a * b, mode: ConvMode.Valid);
            AssertArray.AreAlmostEqual(x_k, x_k1);
        }

        [TestMethod]
        public void CorrelateCustomAgreesWithCorrelate()
        {
            int n = 10;
            var x = NN.Random.Uniform(-1f, 1f, n);

            int n_k = 3;
            var k = NN.Random.Uniform(-1f, 1f, n_k);

            var x_k = x.Correlate(k, mode: ConvMode.Valid);
            var x_k1 = x.CorrelateCustom(k, NN.Zeros(x_k.Shape), (a, b) => a * b, mode: ConvMode.Valid);
            AssertArray.AreAlmostEqual(x_k, x_k1);

            x_k = x.Correlate(k, mode: ConvMode.Same);
            x_k1 = x.CorrelateCustom(k, NN.Zeros(x_k.Shape), (a, b) => a * b, mode: ConvMode.Same);
            AssertArray.AreAlmostEqual(x_k, x_k1);

            x_k = x.Correlate(k, mode: ConvMode.Full);
            x_k1 = x.CorrelateCustom(k, NN.Zeros(x_k.Shape), (a, b) => a * b, mode: ConvMode.Full);
            AssertArray.AreAlmostEqual(x_k, x_k1);
        }
    }
}
