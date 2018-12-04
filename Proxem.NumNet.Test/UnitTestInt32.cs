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

using Proxem.NumNet.Int32;

namespace Proxem.NumNet.Test
{
    [TestClass]
    public class UnitTestInt32
    {
        [TestMethod]
        public void TestMin()
        {
            var a = NN.Range(4 * 2).Reshape(4, 2);
            a.Item[2, 1] = -1;
            a.Item[0, 1] = 10;
            AssertArray.AreEqual(NN.Array(new[] { 0, -1 }), a.Min(axis: 0));
            AssertArray.AreEqual(NN.Array(new[] { 0, 2, -1, 6}), a.Min(axis: 1));
        }

        [TestMethod]
        public void TestMax()
        {
            var a = NN.Range(4 * 2).Reshape(4, 2);
            a.Item[2, 1] = -1;
            a.Item[0, 1] = 10;
            AssertArray.AreEqual(NN.Array(new[] { 6, 10 }), a.Max(axis: 0));
            AssertArray.AreEqual(NN.Array(new[] { 10, 3, 4, 7 }), a.Max(axis: 1));
        }

        [TestMethod]
        public void ArgmaxWorksOnVec()
        {
            Assert.AreEqual(2, NN.Array(0, 1, 5, 1, 0).Argmax());

            AssertArray.AreEqual(
                NN.Array(2),
                NN.Array(0, 1, 5, 1, 0).Argmax(axis: 0, keepDims: true));
        }

        [TestMethod]
        public void ArgmaxWorksOnMatrix()
        {
            var a = NN.Array(new [,]{
                {0, 10},
                {2,  3},
                {4, -1},
                {6,  7}
            });
            AssertArray.AreEqual(NN.Array(new[] { 3, 0 }), a.Argmax(axis: 0));
            AssertArray.AreEqual(NN.Array(new[] { 1, 1, 0, 1 }), a.Argmax(axis: 1));
        }

        [TestMethod]
        public void UnArgmaxWorksOnMatrix()
        {
            var a = NN.Array(new[,]{
                {0, 10},
                {2,  3},
                {4, -1},
                {6,  7}
            });
            var argmax0 = a.Argmax(axis: 0);
            var delta0 = NN.Max(a, axis: 0);
            var selected0 = NN.UnArgmax(delta0, argmax0, 0, a.Shape[0]);
            var expected0 = NN.Array(new[,]{
                {0, 10},
                {0,  0},
                {0,  0},
                {6,  0}
            });
            AssertArray.AreEqual(expected0, selected0);

            var argmax1 = a.Argmax(axis: 1);
            var delta1 = NN.Max(a, axis: 1);
            var selected1 = NN.UnArgmax(delta1, argmax1, 1, a.Shape[1]);
            var expected1 = NN.Array(new[,]{
                {0, 10},
                {0,  3},
                {4,  0},
                {0,  7}
            });
            AssertArray.AreEqual(expected1, selected1);

            argmax1 = a.Argmax(axis: 1, keepDims: true);
            delta1 = NN.Max(a, axis: 1, keepDims: true);
            selected1 = NN.UnArgmax(delta1, argmax1, 1, a.Shape[1], keepDims: true);
            expected1 = NN.Array(new[,]{
                {0, 10},
                {0,  3},
                {4,  0},
                {0,  7}
            });
            AssertArray.AreEqual(expected1, selected1);
        }

        [TestMethod]
        public void ArgmaxWorksOnTensor()
        {
            AssertArray.AreEqual(
                NN.Array(new[,] {
                    { 0, 1, 0, 0 },
                    { 0, 1, 1, 0 }
                }),
                NN.Array(new[,,] {
                    { { 5, 0 }, {0, 5}, { 5, 0 }, { 5, 0 } },
                    { { 5, 0 }, {0, 5}, { 0, 5 }, { 5, 0 } },
                }).Argmax(axis: -1)
            );

            AssertArray.AreEqual(
                NN.Array(new[,] {
                    { 2, 1 },
                    { 0, 1 }
                }),
                NN.Array(new[,,] {
                    { { 1, 0 }, {0, 5}, { 5, 1 }, { 3, 0 } },
                    { { 5, 0 }, {0, 5}, { 0, 5 }, { 1, 0 } },
                }).Argmax(axis: 1)
            );
        }

        [TestMethod]
        public void TestArgmin()
        {
            var a = NN.Range(4 * 2).Reshape(4, 2);
            a.Item[2, 1] = -1;
            a.Item[0, 1] = 10;
            AssertArray.AreEqual(NN.Array(new[] { 0, 2 }), a.Argmin(axis: 0));
            AssertArray.AreEqual(NN.Array(new[] { 0, 0, 1, 0 }), a.Argmin(axis: 1));
        }

        [TestMethod]
        public void TestLtOnInt()
        {
            AssertArray.AreEqual(
                NN.Array(1, 5, 3, 2, 0, -1) < NN.Array(0, 6, 1, 3, 1, 2),
                NN.Array(0, 1, 0, 1, 1, 1)
            );
        }
    }
}
