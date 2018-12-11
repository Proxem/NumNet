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

using Proxem.BlasNet;
using Proxem.NumNet.Single;

namespace Proxem.NumNet.Test
{
    using static Slicer;

    [TestClass]
    public class UnitTestSingle
    {
        [TestMethod]
        public void TestVector()
        {
            var v1 = NN.Array<float>(0, 1, 2);
            var v = new Array<float>(3);

            for (int i = 0; i < 3; i++)
            {
                v.Item[i] = i;
            }
            AssertArray.AreAlmostEqual(v, v1);
        }

        [TestMethod]
        public void TestMatrix()
        {
            var m1 = NN.Array(new float[,] {
                { 0, 1, 2, 3 },
                { 1, 2, 3, 4 },
                { 2, 3, 4, 5 }
            });
            var m = new Array<float>(3, 4);
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 4; j++)
                    m.Item[i, j] = i + j;

            AssertArray.AreAlmostEqual(m, m1);
        }

        [TestMethod]
        public void TestMatrixFromArray()
        {
            var m1 = NN.Array(
                new [] { 0, 1, 2, 3 },
                new [] { 1, 2, 3, 4 },
                new [] { 2, 3, 4, 5 }
            );
            var m2 = NN.Array(new [,]{
                { 0, 1, 2, 3 },
                { 1, 2, 3, 4 },
                { 2, 3, 4, 5 }
            });

            AssertArray.AreEqual(m2, m1);
        }


        [TestMethod]
        public void TestRange()
        {
            var a = NN.Diag(1, 2, 3, 4);
            var b = NN.Diag(2, 3);
            var c = a[(1, 3), (1, 3)];
            AssertArray.AreEqual(c, b);
            c = a[(1, -1), (1, -1)];
            AssertArray.AreEqual(c, b);
        }

        [TestMethod]
        public void TestRange2()
        {
            var range = NN.Range(3, 7);
            Assert.AreEqual("[3 4 5 6]", range.ToString());
            AssertArray.AreEqual(new[] { 3, 4, 5, 6 }, range);
        }

        [TestMethod]
        public void TestDotRight()
        {
            var x = NN.Array<float>(0, -1, 2, 3);

            var m = NN.Array(new float[,] {
                { 0, -1, 2, 3 },
                { 1, -2, 3, 4 },
                { 2, -3, 4, 5 }
            });

            var v = NN.Array<float>(0, -1, 2, 3);

            //Scalar dot
            AssertArray.AreAlmostEqual(14, x.Dot(v));

            //Simple dot
            var result = m.Dot(v);
            var expected = NN.Array<float>(14, 20, 26);
            AssertArray.AreAlmostEqual(expected, result);

            //Blas.gemv(Order.RowMajor, Proxem.BlasNet.Transpose.NoTrans, 4, 3, 1.0f, m.Values, 0, 1, v.Values, 0, 1, 1.0f, result.Values, 0, 1);
            //Blas.gemv(Order.RowMajor, Proxem.BlasNet.Transpose.NoTrans, 4, 3, 1.0f, m.Values, 0, 4, v.Values, 0, 1, 0.0f, result.Values, 0, 1);

            //Dot with slices
            result = m[From(1), Upto(-1)].Dot(v[Upto(-1)]);
            expected = NN.Array<float>(8, 11);
            AssertArray.AreAlmostEqual(expected, result);

            //Dot with transpose
            result = m.T.Dot(NN.Range<float>(3));
            expected = NN.Array<float>(5, -8, 11, 14);
            AssertArray.AreAlmostEqual(expected, result);

            var t = NN.Ones<float>(6, 5, 4);
            t[2, _, _] *= 2;
            t[_, 1, _] *= -1;
            t[_, _, 2] *= 3;

            m = NN.Array(new float[,]{
                {1, -1, 1, 1},
                {2, -2, 2, 2}
            });

            //Matrix dot with stride[1] != 1
            var y = NN.Array<float>(1, -1, 3, 2);
            var my = m.Dot(y);
            var ty = t[Range(1, 3), Upto(-1), -1].Dot(y);
            AssertArray.AreAlmostEqual(t[Range(1, 3), Upto(-1), -1], m);
            AssertArray.AreAlmostEqual(my, NN.Array<float>(7, 14));
            AssertArray.AreAlmostEqual(ty, my);
        }

        [TestMethod]
        public void TestAddVec()
        {
            var a = NN.Array<float>(1, 0, 2, 4);
            var b = NN.Array<float>(2, 1, -1, 3);

            AssertArray.GenerateTests(a, b, (a1, b1) =>
            {
                AssertArray.AreAlmostEqual(NN.Array<float>(3, 1, 1, 7), a1.Add(b1));
                AssertArray.AreAlmostEqual(NN.Array<float>(3, 1, 1, 7), b1.Add(a1));
                AssertArray.AreAlmostEqual(NN.Array<float>(-1, -1, 3, 1), a.Add(b, -1));
                AssertArray.AreAlmostEqual(NN.Array<float>(1, 1, -3, -1), b.Add(a, -1));
            });

            var c = b.Copy();
            AssertArray.AreAlmostEqual(a.Add(c, -1, result: c), NN.Array<float>(-1, -1, 3, 1));

            c = b.Copy();
            c.Acc(a);
            AssertArray.AreAlmostEqual(c, NN.Array<float>(3, 1, 1, 7));

            c = b.Copy();
            AssertArray.AreAlmostEqual(c.Add(a, -1, result: c), NN.Array<float>(1, 1, -3, -1));
        }

        [TestMethod]
        public void TestAddMat()
        {
            var a = NN.Array(new float[,] {
                { 0, -1, 2, 3 },
                { 1, -2, 3, 4 },
                { 2, -3, 4, 5 }
            });

            var b = NN.Array(new float[,] {
                { 5, -1,  1, 3 },
                { 1,  2, -3, 4 },
                { 2,  1,  4, 9 }
            });

            var a_p_b = NN.Array(new float[,] {
                { 5, -2, 3,  6 },
                { 2,  0, 0,  8 },
                { 4, -2, 8, 14 }
            });

            var a_m_b = NN.Array(new float[,] {
                { -5,  0, 1,  0 },
                {  0, -4, 6,  0 },
                {  0, -4, 0, -4 }
            });

            var c = NN.Random.Uniform(-1f, 1f, 6, 3, 10).As<float>();
            var r = c[Range(1, 4), Only(2), Range(9, 1, -2)];
            AssertArray.GenerateTests(a, a1 =>
            {
                AssertArray.AreAlmostEqual(a_p_b, a1.Add(b));
                AssertArray.AreAlmostEqual(a_p_b, b.Add(a1));
                AssertArray.AreAlmostEqual(a_p_b, b.Add(a1, result: r));
                AssertArray.AreAlmostEqual(a_m_b, a.Add(b, -1));
                AssertArray.AreAlmostEqual(-a_m_b, b.Add(a, -1));
            });
        }

        [TestMethod]
        [ExpectedException(typeof(RankException)), TestCategory("Exception")]
        public void FailAccChekShapes()
        {
            NN.Array(1f, 0f, 2f, 4f).Acc(NN.Array(2f, 1f, 3f));
        }

        [TestMethod]
        public void TestMulVec()
        {
            var a = NN.Array<float>(1, 0, 2, 4);
            var b = NN.Array<float>(2, 1, -1, 3);
            var ab = NN.Array<float>(2, 0, -2, 12);

            var c = NN.Random.Uniform(-1f, 1f, 10, 5).As<float>();
            var r = c[Range(9, 1, -2), Only(2)];

            AssertArray.GenerateTests(a, b, (a1, b1) =>
            {
                AssertArray.AreAlmostEqual(ab, a * b);
                AssertArray.AreAlmostEqual(ab, b * a);
                AssertArray.AreAlmostEqual(ab.Reshape(1, -1), a * b.Reshape(1, -1));
                AssertArray.AreAlmostEqual(ab.Reshape(1, -1), a.Reshape(1, -1) * b);
                AssertArray.AreAlmostEqual(ab.Reshape(1, -1), a.Reshape(1, -1) * b.Reshape(1, -1));
                AssertArray.AreAlmostEqual(ab.Reshape(-1, 1), a.Reshape(-1, 1) * b.Reshape(-1, 1));
                AssertArray.AreAlmostEqual(ab, a.Mul(b, result: r));
            });
        }

        [TestMethod]
        public void TestMulMat()
        {
            var a = NN.Array(new float[,] {
                { 0, -1, 2, 3 },
                { 1, -2, 3, 4 },
                { 2, -3, 4, 5 }
            });

            var b = NN.Array(new float[,] {
                { 5, -1,  1, 3 },
                { 1,  2, -3, 4 },
                { 2,  1,  4, 9 }
            });

            var ab = NN.Array(new float[,] {
                { 0,  1,  2,  9 },
                { 1, -4, -9, 16 },
                { 4, -3, 16, 45 }
            });

            var c = NN.Random.Uniform(-1f, 1f, 6, 3, 10).As<float>();
            var r = c[Range(1, 4), Only(2), Range(9, 1, -2)];
            AssertArray.GenerateTests(a, a1 =>
            {
                AssertArray.AreAlmostEqual(ab, a1 * b);
                AssertArray.AreAlmostEqual(ab, b * a1);
                AssertArray.AreAlmostEqual(ab, a1.Mul(b, result: r));
            });
        }

        [TestMethod]
        public void TestTensorDot()
        {
            var a = NN.Range<float>(0, 5);
            var b = NN.Range<float>(5, 10);

            var outer = a.Outer(b);
            var expected = NN.Array(new float[,]{
                {0, 0, 0, 0, 0},
                {5, 6, 7, 8, 9},
                {10, 12, 14, 16, 18},
                {15, 18, 21, 24, 27},
                {20, 24, 28, 32, 36}
            });
            AssertArray.AreAlmostEqual(expected, outer);

            var outer2 = outer[_, 2];
            var expected2 = NN.Array<float>(0, 7, 14, 21, 28);
            AssertArray.AreAlmostEqual(expected2, outer2);

            var outer3 = outer[_, Range(2, 3)];
            var expected3 = NN.Array(new float[,]{
                {0},
                {7},
                {14},
                {21},
                {28}
            });
            AssertArray.AreAlmostEqual(expected3, outer3);

            var tdr = ArrayExtensions.TensorDotRight(outer, b);
            AssertArray.AreAlmostEqual(tdr, outer.TensorDot(b));

            var tdr2 = NN.Ones<float>(tdr.Shape);
            var result4 = outer.TensorDot(b, result: tdr2, beta: 1);    // 5 x 5 x 5
            var slice0 = NN.Array(new float[,]{
                {1, 1, 1, 1, 1},
                {1, 1, 1, 1, 1},
                {1, 1, 1, 1, 1},
                {1, 1, 1, 1, 1},
                {1, 1, 1, 1, 1}
            });
            AssertArray.AreAlmostEqual(slice0, result4[0]);

            var slice1 = NN.Array(new float[,]{
                {26, 31, 36, 41, 46},
                {31, 37, 43, 49, 55},
                {36, 43, 50, 57, 64},
                {41, 49, 57, 65, 73},
                {46, 55, 64, 73, 82}
            });
            AssertArray.AreAlmostEqual(slice1, result4[1]);

            var slice2 = NN.Array(new float[,]{
                { 51, 61, 71, 81, 91 },
                { 61, 73, 85, 97, 109 },
                { 71, 85, 99, 113, 127 },
                { 81, 97, 113, 129, 145 },
                { 91, 109, 127, 145, 163 }
            });
            AssertArray.AreAlmostEqual(slice2, result4[2]);

            var slice3 = NN.Array(new float[,]{
                {76, 91, 106, 121, 136},
                {91, 109, 127, 145, 163},
                {106, 127, 148, 169, 190},
                {121, 145, 169, 193, 217},
                {136, 163, 190, 217, 244}
            });
            AssertArray.AreAlmostEqual(slice3, result4[3]);

            var slice4 = NN.Array(new float[,]{
                {101, 121, 141, 161, 181},
                {121, 145, 169, 193, 217},
                {141, 169, 197, 225, 253},
                {161, 193, 225, 257, 289},
                {181, 217, 253, 289, 325}
            });
            AssertArray.AreAlmostEqual(slice4, result4[4]);

            var tdl = ArrayExtensions.TensorDotLeft(b, outer);
            AssertArray.AreAlmostEqual(tdl, b.TensorDot(outer));
        }

        [TestMethod]
        public void TestTensorDot3Dx2D()
        {
            // a.Shape = [2, 5, 3]
            var a = NN.Array(new float[,,]{
                {{ 0, -1,  0}, { 2, -1,  0}, { 2, -1,  1}, { 0, -1,  0}, {-1,  1, -1}},
                {{ 1, -1,  1}, { 1,  2,  2}, { 1,  0, -1}, { 1,  1,  1}, { 2,  2, -1}}
            });

            // b.Shape = [2, 3]
            var b = NN.Array(new float[,]{{ 2,  2,  1}, { 2,  2, -1}});

            // np.tensordot(a, b, axes=([0, -1], [0, 1]))
            var a02_b01 = NN.TensorDot(a, new int[] { 0, -1 }, b, new int[] { 0, 1 });
            var a02_b01_exp = NN.Array(new float[] { -3,  6,  6,  1,  8 });
            AssertArray.AreAlmostEqual(a02_b01_exp, a02_b01);

            // np.tensordot(a, b, axes=([0], [0]))
            var a0_b0 = NN.TensorDot(a, new int[]{0}, b, new int[]{0});
            var a0_b0_exp = NN.Array(new float[,,]{
                {{ 2,  2, -1}, {-4, -4,  0}, { 2,  2, -1}},
                {{ 6,  6,  1}, { 2,  2, -3}, { 4,  4, -2}},
                {{ 6,  6,  1}, {-2, -2, -1}, { 0,  0,  2}},
                {{ 2,  2, -1}, { 0,  0, -2}, { 2,  2, -1}},
                {{ 2,  2, -3}, { 6,  6, -1}, {-4, -4,  0}}
            });
            AssertArray.AreAlmostEqual(a0_b0_exp, a0_b0);
        }


        [TestMethod]
        public void TestTensorDot2Dx3D()
        {
            // a.Shape = [4, 2, 5]
            var a = NN.Array(new float[,,] {
                {{0, 1, 2, 3, 4}, {2, 3, 4, 5, 6}},
                {{0, 1, 2, 3, 4}, {2, 3, 4, 5, 6}},
                {{0, 1, 2, 3, 4}, {2, 3, 4, 5, 6}},
                {{0, 1, 2, 3, 4}, {2, 3, 4, 5, 6}},
            });

            // b.Shape = [5, 3]
            var b = NN.Array(new float[,] {
                {-1, 1, 1},
                {-1,  2,  4},
                { 1,  3,  3},
                {-1,  1,  3},
                { 0, -1,  4}
            });

            var c = NN.Array(new float[,,] {
                { {-2, 7, 35}, {-6, 19, 65} },
                { {-2, 7, 35}, {-6, 19, 65} },
                { {-2, 7, 35}, {-6, 19, 65} },
                { {-2, 7, 35}, {-6, 19, 65} }
            });

            // np.tensordot(a, b, axes=([0, -1], [0, 1]))
            var a2_b0 = NN.TensorDot(a, new [] { 2 }, b, new[] { 0 });

            var aT = NN.Transpose(a, 1, 0, 2).Copy(); // [2, 4, 5]
            var big = NN.Random.Uniform(-1f, 1f, 10, 4, 5);
            big[From(-2)] = aT;

            var aTT = NN.Transpose(big[From(-2)], 1, 0, 2);

            var aTT2_b0 = NN.TensorDot(aTT, new [] { 2 }, b, new[] { 0 });
            AssertArray.AreAlmostEqual(c, aTT2_b0);
        }

        [TestMethod]
        public void TestTensorDot4Dx3D()
        {
            // a.Shape = [2, 4, 5, 3]
            var a = NN.Array(new float[,,,]{
                {{{ 1,  2,  2}, { 1,  2,  1}, { 0, -1,  0}, { 0, -1,  2}, { 1,  1, -1}}, {{ 1, -1,  2}, { 0, -1,  2}, {-1,  2, -1}, { 0,  2,  1}, {-1,  0,  1}}, {{ 1, -1,  0}, { 0,  1, -1}, {-1,  0, -1}, { 2,  0,  1}, { 2, -1,  2}}, {{ 2,  0, -1}, { 0,  1,  0}, { 2, -1,  2}, { 1,  2,  1}, { 1,  2,  0}}},
                {{{ 0,  2,  0}, { 0,  2,  2}, { 0,  1,  1}, { 2,  0, -1}, {-1,  2,  1}}, {{ 2, -1,  2}, { 0, -1, -1}, { 0, -1,  1}, { 0, -1, -1}, {-1,  0,  1}}, {{-1,  2,  0}, {-1,  2, -1}, { 2,  1,  1}, {-1,  1,  1}, {-1,  1, -1}}, {{-1,  1,  0}, { 2,  1,  2}, { 2,  1,  0}, { 0,  0, -1}, {-1,  1,  1}}}
            });

            // b.Shape = [2, 3, 5]
            var b = NN.Array(new float[,,]{
                {{-1,  0,  2,  1, -1}, { 1,  1,  0,  1,  2}, { 0,  1, -1,  2, -1}},
                {{ 0, -1,  0, -1, -1}, { 2,  2, -1,  1, -1}, {-1,  2,  1,  2, -1}}
            });

            // np.tensordot(a, b, axes=([0, -1, 2], [0, 1, 2]))
            var a032_b012 = NN.TensorDot(a, new int[] { 0, 3, 2 }, b, new int[] { 0, 1, 2 });
            var a032_b012_exp = NN.Array(new float[] { 15, -7, 7, 11 });
            AssertArray.AreAlmostEqual(a032_b012_exp, a032_b012);

            // np.tensordot(a, b, axes=([0, -1], [0, 1]))
            var a03_b01 = NN.TensorDot(a, new int[] { 0, 3 }, b, new int[] { 0, 1 });
            var a03_b01_exp = NN.Array(new float[,,]{
                { {  5,  8, -2,  9, -1}, {  3, 11,  1, 11, -2}, {  0,  3,  0,  2, -4}, {  0, -3, -3, -1, -5}, { 3,  7,  2,  5,  0} },
                { { -6,  1,  3,  5, -8}, { -2, -3, -2,  0, -2}, {  0,  1,  1,  0,  6}, {  1, -1, -1,  1,  5}, { 0,  4, -2,  4,  0} },
                { {  2,  4,  0,  3, -4}, {  6,  3, -2,  0,  3}, {  2,  1, -1, -2, -2}, { -1,  6,  3,  8, -4}, { 0,  2,  0,  5, -5} },
                { {  0,  2,  4,  2, -1}, {  1,  5,  1,  4, -3}, { -1,  1,  1,  4, -9}, {  2,  1,  0,  3,  3}, { 2,  7,  2,  7,  2} }
            });
            AssertArray.AreAlmostEqual(a03_b01_exp, a03_b01);
        }

        [TestMethod]
        public void TestDotLeft()
        {
            var m = NN.Array(new float[,] {
                { 0, 1, 2, 3 },
                { 1, 2, 3, 4 },
                { 2, 3, 4, 5 }
            });

            var v = NN.Range<float>(3);
            var result = v.Dot(m);
            var expected = NN.Array<float>(5, 8, 11, 14);
            AssertArray.AreAlmostEqual(expected, result);

            result = m.Dot(v, transA: true);
            AssertArray.AreAlmostEqual(expected, result);
        }

        [TestMethod]
        public void TestVectorDot()
        {
            var v1 = NN.Range<float>(4);
            var v2 = NN.Range<float>(1, 5);

            Assert.AreEqual(v1.VectorDot(v2), 20f);
            Assert.AreEqual(v2.VectorDot(v1), 20f);
        }

        [TestMethod]
        public void TestNorm()
        {
            var v1 = NN.Range<float>(4);
            Assert.AreEqual((float)Math.Sqrt(0 * 0 + 1 * 1 + 2 * 2 + 3 * 3), NN.Norm(v1));
        }

        [TestMethod]
        public void TestOuter()
        {
            var a = NN.Ones<float>(4);
            var b = NN.Ones<float>(3);
            a.Item[2] = 3;
            b.Item[1] = 2;

            var expected = NN.Array(new float[,]{
                { 1, 1, 3, 1 },
                { 2, 2, 6, 2 },
                { 1, 1, 3, 1 }
            });

            AssertArray.AreAlmostEqual(expected, b.Outer(a));
        }

        [TestMethod]
        public void TestOuter2()
        {
            var a = NN.Ones<float>(5, 5);
            var b = NN.Ones<float>(4, 4);
            a.Item[0, 2] = 3;
            b.Item[0, 3] = 2;

            var c = a.Outer(b);
            //var expected = NN.Array(new float[,]{
            //    { 1, 1, 3, 1 },
            //    { 2, 2, 6, 2 },
            //    { 1, 1, 3, 1 }
            //});

            var expected = NN.Array(new float[,]{
                {1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                {1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                {3, 3, 3, 6, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3},
                {1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                {1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                {1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                {1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                {1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                {1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                {1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                {1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                {1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                {1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                {1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                {1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                {1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                {1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                {1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                {1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                {1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                {1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                {1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                {1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                {1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                {1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
            });
            AssertArray.AreAlmostEqual(c, expected);
        }



        [TestMethod]
        public void TestScale()
        {
            var x = NN.Const(0.5f, 5, 4);
            var x2 = NN.Ones<float>(5, 4).Scale(0.5f);
            AssertArray.AreAlmostEqual(x, x2);

            var y = NN.Const(0.3f, 10);
            var y2 = NN.Ones<float>(10);
            y2.Scale(0.3f, result: y2);
            AssertArray.AreAlmostEqual(y, y2);
        }

        [TestMethod]
        public void TestTranspose()
        {
            var t = NN.Ones<float>(5, 4, 3);
            t.AssertOfShape(5, 4, 3);

            t.Transpose(0, 1, 2).AssertOfShape(5, 4, 3);
            t.Transpose(0, 2, 1).AssertOfShape(5, 3, 4);

            t.Transpose(1, 0, 2).AssertOfShape(4, 5, 3);
            t.Transpose(1, 2, 0).AssertOfShape(4, 3, 5);

            t.Transpose(2, 0, 1).AssertOfShape(3, 5, 4);
            t.Transpose(2, 1, 0).AssertOfShape(3, 4, 5);
        }

        [TestMethod]
        public void TestBroadcast()
        {
            var w = NN.Ones<float>(2, 2);
            var b = NN.Ones<float>(2, 1);
            b.Item[1, 0] = 2;

            var result = w + b;

            var expected = NN.Array(new float[,] {
                { 2, 2 },
                { 3, 3 }
            });

            AssertArray.AreAlmostEqual(expected, result);

            b = NN.Ones<float>(2);
            b.Item[1] = 2;

            result = w + b;

            expected = NN.Array(new float[,] {
                { 2, 3 },
                { 2, 3 }
            });

            AssertArray.AreAlmostEqual(expected, result);
        }

        [TestMethod]
        public void TestBroadcast2()
        {
            var a = NN.Ones<float>(5, 3);
            var b = NN.Ones<float>(1, 3);
            var c = NN.Ones<float>(1, 1);

            for (int i = 0; i < 5; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    a.Item[i, j] = i + j;
                }
            }

            b.Item[0, 0] = 2;
            b.Item[0, 1] = 3;
            b.Item[0, 2] = 4;

            c.Item[0, 0] = 2;

            var aExpected = NN.Array(new float[,]{
                { 0, 1, 2 },
                { 1, 2, 3 },
                { 2, 3, 4 },
                { 3, 4, 5 },
                { 4, 5, 6 }
            });

            AssertArray.AreAlmostEqual(a, aExpected);

            var bExpected = NN.Array(new float[,]{
                { 2, 3, 4 }
            });

            AssertArray.AreAlmostEqual(b, bExpected);

            var cExpected = NN.Array(new float[,]{
                { 2 }
            });

            AssertArray.AreAlmostEqual(c, cExpected);

            var d = a * b;
            var dExpected = NN.Array(new float[,]{
                { 0, 3, 8 },
                { 2, 6, 12 },
                { 4, 9, 16 },
                { 6, 12, 20 },
                { 8, 15, 24 },
            });
            AssertArray.AreAlmostEqual(d, dExpected);

            var e = a * c;
            var eExpected = NN.Array(new float[,]{
                { 0, 2, 4 },
                { 2, 4, 6 },
                { 4, 6, 8 },
                { 6, 8, 10 },
                { 8, 10, 12 },
            });
            AssertArray.AreAlmostEqual(e, eExpected);
        }



        [TestMethod]
        public void SanityTest()
        {
            int n = 2;
            float[] x = new[] { -0.43325308f, -0.4298405f };
            int offsetx = 0;
            int incx = 1;
            float[] y = new[] { 0f, 1f };
            int offsety = 0;
            int incy = 1;
            var result = Blas.dot(n, x, offsetx, incx, y, offsety, incy);
            Console.WriteLine(result);

            var a = NN.Array(x);
            var b = NN.Array(y);
            var c = (float)a.T.Dot(b);
            Console.WriteLine(c);
            AssertArray.AreAlmostEqual(x[1], c);
            AssertArray.AreAlmostEqual(x[1], result);
        }

        [TestMethod]
        public void TestDot1()
        {
            var m = new Array<float>(3, 4);
            var v = new Array<float>(3);

            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 4; j++)
                    m.Item[i, j] = i + j;
                v.Item[i] = i;
            }

            var result = v.Dot(m);
            AssertArray.AreAlmostEqual(result, NN.Array<float>(5, 8, 11, 14));
        }

        [TestMethod]
        public void TestDot2()
        {
            var m = new Array<float>(4, 3);
            var v = new Array<float>(3);

            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 4; j++)
                    m.Item[j, i] = i + j;
                v.Item[i] = i;
            }

            var result = m.Dot(v);
            AssertArray.AreAlmostEqual(result, NN.Array<float>(5, 8, 11, 14));
        }

        [TestMethod]
        public void TestDot3()
        {
            var a = NN.Range<float>(4);
            var b = NN.Range<float>(3 * 4 * 5).Reshape(3, 4, 5);
            var c = a.Dot(b);

            var expected = NN.Array(new float[,] {
                { 70, 76, 82, 88, 94, },
                { 190, 196, 202, 208, 214 },
                { 310, 316, 322, 328, 334 }
            });

            AssertArray.AreAlmostEqual(c, expected);
        }

        [TestMethod]
        public void TestSumAlong()
        {
            var a = NN.Range<float>(3 * 4 * 5).Reshape(3, 4, 5);
            var b = a.Sum(axis: 1);

            var expected = NN.Array(new float[,] {
                { 30,  34,  38,  42,  46},
                {110, 114, 118, 122, 126},
                {190, 194, 198, 202, 206}
            });

            AssertArray.AreAlmostEqual(expected, b);

            var c = a.Sum(axis: -1);
            expected = NN.Array(new float[,] {
                {10,  35,  60,  85},
                {110, 135, 160, 185},
                {210, 235, 260, 285}
            });
            AssertArray.AreAlmostEqual(expected, c);
        }

        [TestMethod]
        public void TestCopy()
        {
            var v0 = NN.Array<float>(2, 2, -1, 2, 3, 0, -1, 6, -1, 5);
            var v = v0.Copy();
            AssertArray.AreEqual(v0, v);
            AssertArray.AreEqual(v0, NN.Copy(v));
            AssertArray.AreEqual(v0, NN.Copy<float>(v));

            v[Range(2, 6)] = v[Upto(4)];
            AssertArray.AreNotEqual(v0, v);

            var v1 = NN.Array<float>(2, 2, 2, 2, -1, 2, -1, 6, -1, 5);
            AssertArray.AreEqual(v1, v);

            v = v0.Copy();
            v[Range(6, 2, -1)] = v[Range(4, null, -1)];
        }

        [TestMethod]
        public void TestApply()
        {
            var v0 = NN.Array<float>(2, 2, -1, 2, 3, 0, -1, 6, -1, 5);
            var v = NN.Apply(v0, x => x);
            AssertArray.AreEqual(v0, v);

            v0.Item[0] = 0;
            AssertArray.AreNotEqual(v0, v);
        }

        [TestMethod]
        public void TestApplyWithResult()
        {
            var v0 = NN.Array<float>(2, 2, -1, 2, 3, 0, -1, 6, -1, 5);
            var v1 = NN.Zeros(v0.Size);
            var v = NN.Apply(v0, x => x, result: v1);
            Assert.AreEqual(v, v1);
            AssertArray.AreEqual(v0, v);
            AssertArray.AreEqual(v0, v1);

            v0.Item[0] = 0;
            AssertArray.AreNotEqual(v0, v);
        }

        [TestMethod]
        public void TestMax1()
        {
            var x = NN.Range<float>(3 * 4 * 5).Reshape(3, 4, 5);

            AssertArray.AreAlmostEqual(new float[,] {
                { 40, 41, 42, 43, 44 },
                { 45, 46, 47, 48, 49 },
                { 50, 51, 52, 53, 54 },
                { 55, 56, 57, 58, 59 } }, NN.Max(x, axis: 0));

            AssertArray.AreAlmostEqual(new float[,] {
                { 15, 16, 17, 18, 19 },
                { 35, 36, 37, 38, 39 },
                { 55, 56, 57, 58, 59 } }, NN.Max(x, axis: 1));

            AssertArray.AreAlmostEqual(new float[,] {
                { 4, 9, 14, 19 },
                { 24, 29, 34, 39 },
                { 44, 49, 54, 59} }, NN.Max(x, axis: 2));
        }

        [TestMethod]
        public void TestMax2()
        {
            var x = NN.Array(new float[,,] { { { 5, 2, 1 }, { 2, 4, 8 } }, { { 7, 1, 9 }, { 2, 9, 4 } } });

            AssertArray.AreAlmostEqual(new float[,] {
                { 7, 2, 9 },
                { 2, 9, 8 } }, NN.Max(x, axis: 0));

            AssertArray.AreAlmostEqual(new float[,] {
                { 5, 4, 8 },
                { 7, 9, 9 } }, NN.Max(x, axis: 1));

            AssertArray.AreAlmostEqual(new float[,] {
                { 5, 8 },
                { 9, 9 } }, NN.Max(x, axis: 2));
        }

        [TestMethod]
        public void TestInsert()
        {
            var a = NN.Array(new[,] { { 1, 1 }, { 2, 2 }, { 3, 3 } });
            AssertArray.AreEqual(a.Insert(NN.Array(new[] { 0, 0 }), 0, 0), NN.Array(new[,] { { 0, 0 }, { 1, 1 }, { 2, 2 }, { 3, 3 } }));
            AssertArray.AreEqual(a.Insert(NN.Array(new[] { 4, 4 }), int.MaxValue, 0), NN.Array(new[,] { { 1, 1 }, { 2, 2 }, { 3, 3 }, { 4, 4 } }));
        }

        [TestMethod]
        public void TestMin()
        {
            var a = NN.Range<float>(4 * 2).Reshape(4, 2);
            a.Item[2, 1] = -1;
            a.Item[0, 1] = 10;
            AssertArray.AreAlmostEqual(NN.Array(new[] { 0f, -1f }), a.Min(axis: 0));
            AssertArray.AreAlmostEqual(NN.Array(new[] { 0f, 2f, -1f, 6f }), a.Min(axis: 1));
        }

        [TestMethod]
        public void TestMax()
        {
            var a = NN.Range<float>(4 * 2).Reshape(4, 2);
            a.Item[2, 1] = -1;
            a.Item[0, 1] = 10;
            AssertArray.AreAlmostEqual(NN.Array(new[] { 6f, 10f }), a.Max(axis: 0));
            AssertArray.AreAlmostEqual(NN.Array(new[] { 10f, 3f, 4f, 7f }), a.Max(axis: 1));
        }

        [TestMethod]
        public void TestArgmax()
        {
            var a = NN.Range<float>(4 * 2).Reshape(4, 2);
            a.Item[2, 1] = -1;
            a.Item[0, 1] = 10;
            AssertArray.AreEqual(NN.Array(new[] { 3, 0 }), a.Argmax(axis: 0));
            AssertArray.AreEqual(NN.Array(new[] { 1, 1, 0, 1 }), a.Argmax(axis: 1));
        }

        [TestMethod]
        public void TestArgmin()
        {
            var a = NN.Range<float>(4 * 2).Reshape(4, 2);
            a.Item[2, 1] = -1;
            a.Item[0, 1] = 10;
            AssertArray.AreEqual(NN.Array(new[] { 0, 2 }), a.Argmin(axis: 0));
            AssertArray.AreEqual(NN.Array(new[] { 0, 0, 1, 0 }), a.Argmin(axis: 1));
        }

        [TestMethod]
        public void TestCov()
        {
            {
                var x = NN.Array(new float[,] { { 0, 2 }, { 1, 1 }, { 2, 0 } });
                var expected = new float[,] { { 1, -1 }, { -1, 1 } };
                AssertArray.AreEqual(expected, NN.Cov(x.T));

                x = NN.Array(new float[,] { { 0, 2 }, { 1, 1 }, { 2, 0 }, { -1, 1 } });
                expected = new float[,] { { 5f / 3, -2f / 3 }, { -2f / 3, 2f / 3 } };
                AssertArray.AreAlmostEqual(expected, NN.Cov(x.T));
            }

            {
                var x = NN.Array(new double[,] { { 0, 2 }, { 1, 1 }, { 2, 0 } });
                var expected = new double[,] { { 1, -1 }, { -1, 1 } };
                AssertArray.AreEqual(expected, NN.Cov(x.T));

                x = NN.Array(new double[,] { { 0, 2 }, { 1, 1 }, { 2, 0 }, { -1, 1 } });
                expected = new double[,] { { 5.0 / 3, -2.0 / 3 }, { -2.0 / 3, 2.0 / 3 } };
                AssertArray.AreAlmostEqual(NN.Array(expected), NN.Cov(x.T));
            }
        }


        [TestMethod]
        public void TestTile()
        {
            var a = NN.Array(new float[] { 0, 1, 2 });
            AssertArray.AreEqual(new float[] { 0, 1, 2, 0, 1, 2 }, NN.Tile(a, 2));
            AssertArray.AreEqual(new float[,] { { 0, 1, 2, 0, 1, 2 }, { 0, 1, 2, 0, 1, 2 } }, NN.Tile(a, 2, 2));
            AssertArray.AreEqual(new float[,,] { { { 0, 1, 2, 0, 1, 2 } }, { { 0, 1, 2, 0, 1, 2 } } }, NN.Tile(a, 2, 1, 2));

            var b = NN.Array(new float[,] { { 1, 2 }, { 3, 4 } });
            // TODO:
            //AssertArray.AreEqual(new float[,] { { 1, 2, 1, 2 }, { 3, 4, 3, 4 } }, NN.Tile(b, 2));
            AssertArray.AreEqual(new float[,] { { 1, 2 }, { 3, 4 }, { 1, 2 }, { 3, 4 } }, NN.Tile(b, 2, 1));
       }

        [TestMethod]
        public void TestLtOnFloat()
        {
            AssertArray.AreEqual(
                NN.Array<float>(1, 5, 3, 2, 0, -1) < NN.Array<float>(0, 6, 1, 3, 1, 2),
                NN.Array<float>(0, 1, 0, 1, 1, 1)
            );
        }

        [TestMethod]
        public void TestRowNormOnFloat()
        {
            Assert.AreEqual(1f, NN.RowNorm(NN.Eye(5)));
        }

        [TestMethod]
        public void ApplyWorksWithParrallelism()
        {
            for (int n = 1000; n < 1005; ++n)
            {
                var x = NN.Random.Uniform(-1f, 1f, n);
                Blas.NThreads = 1;
                var y = NN.Tanh(x);
                Blas.NThreads = 4;
                var z = NN.Tanh(x);
                AssertArray.AreEqual(y, z);
            }
        }

        [TestMethod]
        public void Apply2WorksWithParrallelism()
        {
            int threads = Blas.NThreads;
            for (int n = 1000; n < 1005; ++n)
            {
                var x0 = NN.Random.Uniform(-1f, 1f, n);
                var x1 = NN.Random.Uniform(-1f, 1f, n);
                Blas.NThreads = 1;
                var y = NN.Apply(x0, x1, (_x0, _x1) => _x0 + _x1 * _x1);
                Blas.NThreads = 4;
                var z = NN.Apply(x0, x1, (_x0, _x1) => _x0 + _x1 * _x1);
                AssertArray.AreEqual(y, z);
            }
            Blas.NThreads = threads;
        }

        [TestMethod]
        public void TestDirichlet()
        {
            // with double precision
            AssertArray.AreAlmostEqual(NN.DirichletPdf(0.1, new[] { 0.1, 0.9 }), 0.44298895866652338);
            AssertArray.AreAlmostEqual(NN.DirichletPdf(0.1, new[] { 0.25, 0.25, 0.25, 0.25 }), 0.039814938846965957);
            AssertArray.AreAlmostEqual(NN.DirichletPdf(0.1, new[] { 0.21, 0.25, 0.25, 0.29 }), 0.040755150877577168);
            AssertArray.AreAlmostEqual(NN.DirichletPdf(0.1, new[] { 0.25, 0.21, 0.29, 0.25 }), 0.040755150877577168);
            AssertArray.AreAlmostEqual(NN.DirichletPdf(1.1, new[] { 0.21, 0.25, 0.25, 0.29 }), 7.0885596941973388);


            // with float precision
            AssertArray.AreAlmostEqual(NN.DirichletPdf(0.1f, new[] { 0.25f, 0.25f, 0.25f, 0.25f }), 0.039814938846965957f);
            AssertArray.AreAlmostEqual(NN.DirichletPdf(0.1f, new[] { 0.25f, 0.21f, 0.29f, 0.25f }), 0.040755150877577168f);
            AssertArray.AreAlmostEqual(NN.DirichletPdf(1.1f, new[] { 0.21f, 0.25f, 0.25f, 0.29f }), 7.0885596941973388f);
        }

    }
}
