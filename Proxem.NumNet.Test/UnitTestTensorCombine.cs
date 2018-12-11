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
    public class UnitTestTensorCombine
    {
        [TestMethod]
        public void TestCombine()
        {
            var t = NN.Ones<float>(5, 4, 3);
            t[2, _, _] *= 2;
            t[_, 1, _] *= -1;
            t[_, _, 2] *= 3;

            var x = NN.Array<float>(1, -1, 3);
            var y = NN.Array<float>(1, -1, 3, 2);
            var txy = t.Combine(x, y);

            var z = NN.Zeros<float>(5);
            for (int k = 0; k < z.Shape[0]; ++k)
                for (int j = 0; j < y.Shape[0]; ++j)
                    for (int i = 0; i < x.Shape[0]; ++i)
                        z.Item[k] += t.Item[k, j, i] * y.Item[j] * x.Item[i];

            var expected = new float[] { 63, 63, 126, 63, 63 };
            AssertArray.AreAlmostEqual(expected, z);
            AssertArray.AreAlmostEqual(expected, t.Dot(x).Dot(y));
            AssertArray.AreAlmostEqual(expected, txy);
        }

        [TestMethod]
        public void TestCombine2()
        {
            var a = NN.Ones<float>(4, 5, 6);
            var x = NN.Ones<float>(6);
            var y = NN.Ones<float>(5);
            var z = a.Combine(x, y);

            var expected = NN.Array<float>(30, 30, 30, 30);

            AssertArray.AreAlmostEqual(expected, z);
        }


        [TestMethod]
        public void TestCombineWithBias3D_1D()
        {
            var t = NN.Ones<float>(6, 5, 4);
            t[2, _, _] *= 2;
            t[_, 1, _] *= -1;
            t[_, _, 2] *= 3;

            var x = NN.Array<float>(1, -1, -2);
            var y = NN.Array<float>(1, -1, 3, 1);

            Array<float> txy2 = null;
            txy2 = t[_, Upto(-1), Upto(-1)].Combine(x, y, result: txy2);

            var txy = t.CombineWithBias(x, y);

            var xb = NN.Ones<float>(4);
            xb[Upto(-1)] = x;
            var yb = NN.Ones<float>(5);
            yb[Upto(-1)] = y;

            var txbyb = t.Combine(xb, yb);

            AssertArray.AreAlmostEqual(txbyb, txy);
        }



        [TestMethod]
        public void TestCombineTransposedVersion()
        {
            float r = 0.5f;
            var t = NN.Random.Uniform(-r, r, 5, 4, 3).As<float>();

            var x = NN.Random.Uniform(-r, r, 3).As<float>();
            var y = NN.Random.Uniform(-r, r, 4).As<float>();
            var z = NN.Random.Uniform(-r, r, 5).As<float>();

            var txy = t.Combine21(x, y);
            var txy_ = t.Dot(x).Dot(y);
            AssertArray.AreAlmostEqual(txy_, txy);

            var ztx = t.Combine20(x, z);
            var ztx_ = z.Dot(t.Dot(x));
            AssertArray.AreAlmostEqual(ztx_, ztx);

            var zyt = t.Combine10(y, z);
            var zyt_ = z.Dot(y.Dot(t));
            AssertArray.AreAlmostEqual(zyt_, zyt);
        }

        [TestMethod]
        public void TestCombineDispatchOnTranspose()
        {
            float r = 0.5f;
            var t = NN.Random.Uniform(-r, r, 5, 4, 3).As<float>();

            var x = NN.Random.Uniform(-r, r, 3).As<float>();
            var y = NN.Random.Uniform(-r, r, 4).As<float>();
            var z = NN.Random.Uniform(-r, r, 5).As<float>();

            var txy = t.Transpose(0, 1, 2).Combine(x, y);
            var tyx = t.Transpose(0, 2, 1).Combine(y, x);
            var txy_ = t.Dot(x).Dot(y);
            AssertArray.AreAlmostEqual(txy, tyx);

            var txz = t.Transpose(1, 0, 2).Combine(x, z);
            var tzx = t.Transpose(1, 2, 0).Combine(z, x);
            var ztx = z.Dot(t.Dot(x));
            AssertArray.AreAlmostEqual(txz, tzx);

            var tyz = t.Transpose(2, 0, 1).Combine(y, z);
            var tzy = t.Transpose(2, 1, 0).Combine(z, y);
            var zyt = z.Dot(y.Dot(t));
            AssertArray.AreAlmostEqual(tyz, tzy);
        }
    }
}
