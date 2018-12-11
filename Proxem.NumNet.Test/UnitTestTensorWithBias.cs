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
using Proxem.NumNet.Single;

namespace Proxem.NumNet.Test
{
    using static Slicer;

    [TestClass]
    public class UnitTestTensorWithBias
    {
        [TestMethod]
        public void TestDotWithBias()
        {
            var a = NN.Zeros<float>(3, 4);
            var id = NN.Eye<float>(3);
            a[_, Upto(-1)] = id;

            var expected = NN.Array<float>(new float[,]{
                { 1, 0, 0, 0 },
                { 0, 1, 0, 0 },
                { 0, 0, 1, 0 }
            });

            AssertArray.AreAlmostEqual(expected, a);
            //var x = Tensor.Ones(3);
            //var y = Tensor.Ones(3).Scale(2);
            //Assert.AreEqual(x, y);
        }

        [TestMethod]
        public void TestDotWithBias1D_1D()
        {
            var a = NN.Array<float>(1, 2, 1, 3);
            var b = NN.Array<float>(1, -1, 1);

            AssertArray.AreAlmostEqual(3, (float)a.DotWithBias(b));

            a = NN.Random.Uniform(-1, 1, 5).As<float>();
            b = NN.Random.Uniform(-1, 1, 4).As<float>();
            var c = NN.Ones<float>(5);
            c[Upto(4)] = b;

            AssertArray.AreAlmostEqual(a.Dot(c), a.DotWithBias(b));
        }

        [TestMethod]
        public void TestDotWithIdentity()
        {
            var a = NN.Ones<float>(4, 5);
            a[_, Upto(-1)] = NN.Eye<float>(4);
            var b = NN.Random.Uniform(-1, 1, 4).As<float>();
            var c = NN.Ones<float>(5);
            c[Upto(4)] = b;

            var ac = a.Dot(c);
            var ab = a.DotWithBias(b);

            AssertArray.AreEqual(ac, ab);
        }

        [TestMethod]
        public void TestDotWithBias2D_1D()
        {
            var a = NN.Array<float>(new float[,]{
                { 1, 1, 3, 1 },
                { 2, 2, 6, 2 },
                { 1, 1, 3, 1 }
            });
            var b = NN.Array<float>(-1, 1, 4);
            var c = NN.Ones<float>(4);
            c[Upto(-1)] = b;

            var ac = a.Dot(c);
            var ab = a.DotWithBias(b);
            AssertArray.AreEqual(ac, ab);
        }

        [TestMethod]
        public void TestCombineWithBias3D_1D()
        {
            var t = NN.Ones<float>(6, 5, 4);
            t[2, _, _] *= 2;
            t[_, 1, _] *= -1;
            t[_, _, 2] *= 3;

            var x = NN.Array<float>(1, 2, -2);
            var y = NN.Array<float>(1, -1, 3, 1);

            var txy = t.CombineWithBias(x, y);

            var xb = NN.Ones<float>(4);
            xb[Upto(-1)] = x;
            var yb = NN.Ones<float>(5);
            yb[Upto(-1)] = y;

            var txbyb = t.Combine(xb, yb);

            AssertArray.AreEqual(txbyb, txy);
        }
    }
}
