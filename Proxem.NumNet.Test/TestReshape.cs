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

namespace Proxem.NumNet.Test
{
    using static Slice;

    [TestClass]
    public class TestReshape
    {
        [TestMethod]
        public void TestSimpleReshape()
        {
            var a = NN.Range(4 * 3).Reshape(4, 3);
            var exp = NN.Array(new int[,]{
                {0,  1,  2},
                {3,  4,  5},
                {6,  7,  8},
                {9, 10, 11}
            });

            AssertArray.AreEqual(exp, a);
            AssertArray.AreEqual(exp, NN.Range(4 * 3).Reshape(4, -1));
            AssertArray.AreEqual(exp, NN.Range(4 * 3).Reshape(-1, 3));
            AssertArray.AreEqual(NN.Range(12), a.Reshape(-1));
        }

        [TestMethod, TestCategory("Exception"), ExpectedException(typeof(ArgumentException))]
        public void ComplexReshapeFailsWithoutCopyFlag()
        {
            var a = NN.Range(4 * 3).Reshape(4, 3);

            a.T.Reshape(new int[] { -1 }, allowCopy: false);
        }

        [TestMethod]
        public void ComplexReshapeWorksWithCopyFlag()
        {
            var a = NN.Range(4 * 3).Reshape(4, 3);
            var exp = NN.Array(new int[] { 0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11 });

            AssertArray.AreEqual(exp, a.T.Reshape(new int[]{-1}, allowCopy: true));
        }

        [TestMethod]
        public void CanReshapeReversedArray()
        {
            var a = NN.Zeros<int>(6);
            a[Downward()] = NN.Range(6);
            a = a[Downward()];

            var b = NN.Array(new[,] {
                { 0, 1, 2 },
                { 3, 4, 5 }
            });

            AssertArray.AreEqual(NN.Range(6), a);
            AssertArray.AreEqual(b, a.Reshape(2, 3));
        }

        [TestMethod]
        public void CanReshape_6_to_2_3()
        {
            var a0 = NN.Range(6);
            var b = NN.Array(new[,] {
                { 0, 1, 2 },
                { 3, 4, 5 }
            });

            AssertArray.GenerateTests(a0, a => AssertArray.AreEqual(b, a.Reshape(2, 3)));
        }

        [TestMethod]
        public void CanReshape_6_to_2_3_WithForcedCopy()
        {
            var a0 = NN.Range(6);
            var b = NN.Array(new[,] {
                { 0, 1, 2 },
                { 3, 4, 5 }
            });

            AssertArray.GenerateTests(a0, a => AssertArray.AreEqual(b, a.Reshape(new[] { 2, 3 }, forceCopy: true)));
        }

        [TestMethod]
        public void ConcatWorksWithTwoArrays()
        {
            // Test Concat(1, 2D, 2D)
            var a = NN.Range(12).Reshape(3, 4);
            var b = NN.Range(9).Reshape(3, 3);
            var c = NN.Concat(1, a, b);

            var d = NN.Array(new int[,]
            {
                { 0, 1,  2,  3, /**/ 0, 1, 2 },
                { 4, 5,  6,  7, /**/ 3, 4, 5 },
                { 8, 9, 10, 11, /**/ 6, 7, 8 },
            });
            AssertArray.AreEqual(d, c);

            // Test Concat(0, 2D, 2D)
            a = NN.Range(12).Reshape(4, 3);
            c = NN.Concat(0, a, b);
            d = NN.Array(new int[,]
            {
                { 0,  1,  2 },
                { 3,  4,  5 },
                { 6,  7,  8 },
                { 9, 10, 11 },
                /* -------- */
                { 0,  1,  2 },
                { 3,  4,  5 },
                { 6,  7,  8 }
            });
            AssertArray.AreEqual(d, c);
        }

        [TestMethod]
        public void ConcatWorksWithResult()
        {
            // Test Concat(1, 2D, 2D)
            var a = NN.Range(12).Reshape(3, 4);
            var b = NN.Range(9).Reshape(3, 3);
            var r = NN.Zeros<int>(3, 4 + 3);
            var c = NN.Concat(1, new[] { a, b }, result: r);

            var d = NN.Array(new int[,]
            {
                { 0, 1,  2,  3, /**/ 0, 1, 2 },
                { 4, 5,  6,  7, /**/ 3, 4, 5 },
                { 8, 9, 10, 11, /**/ 6, 7, 8 },
            });
            AssertArray.AreEqual(d, c);

            // Test Concat(0, 2D, 2D)
            a = NN.Range(12).Reshape(4, 3);
            r = NN.Zeros<int>(4 + 3, 3);
            c = NN.Concat(0, new[] { a, b }, result: r);
            d = NN.Array(new int[,]
            {
                { 0,  1,  2 },
                { 3,  4,  5 },
                { 6,  7,  8 },
                { 9, 10, 11 },
                /* -------- */
                { 0,  1,  2 },
                { 3,  4,  5 },
                { 6,  7,  8 }
            });
            AssertArray.AreEqual(d, c);
        }
    }
}
