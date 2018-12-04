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
    public class TestSlicing
    {
        [TestMethod]
        public void TestGenerateVec()
        {
            var a = NN.Range<float>(5);
            var c = a.Copy();
            AssertArray.GenerateTests(a, b => AssertArray.AreAlmostEqual(c, b));
        }

        [TestMethod]
        public void TestGenerateMat()
        {
            var a = NN.Range<float>(12).Reshape(4, 3);
            var c = a.Copy();
            AssertArray.GenerateTests(a, b => AssertArray.AreAlmostEqual(c, b));
        }

        [TestMethod]
        public void TestRange()
        {
            var a = NN.Range(4);
            AssertArray.AreEqual(new[] { 0, 1, 2, 3 }, a);
            AssertArray.AreEqual(new[] { 0, 1, 2, 3 }, a[_]);

            AssertArray.AreEqual(new[] { 0, 1 }, a[Range(0, 2)]);
            AssertArray.AreEqual(new[] { 1 },    a[Range(1, 2)]);
            AssertArray.AreEqual(new[] { 0, 1 }, a[Until(2)]);
            AssertArray.AreEqual(new[] { 2, 3 }, a[From(2)]);

            AssertArray.AreEqual(new[] { 0, 1 },    a[Range(0, -2)]);
            AssertArray.AreEqual(new[] { 1, 2 },    a[Range(1, -1)]);
            AssertArray.AreEqual(new[] { 0, 1, 2 }, a[Until(-1)]);
            AssertArray.AreEqual(new[] { 1, 2, 3 }, a[From(-3)]);

            Assert.AreEqual(2, a.Item[2]);
            Assert.AreEqual(3, a.Item[-1]);
        }

        [TestMethod]
        public void TestSliceLastCol()
        {
            var a = NN.Diag(0, 1, 2, 3);
            a.Item[0, 3] = -1;

            AssertArray.AreEqual(new[] { -1, 0, 0, 3 }, a[_, -1]);
        }

        [TestMethod]
        public void TestSlice1D_in_2D()
        {
            var a = NN.Diag<float>(0, 1, 2, 3);
            var b = NN.Array<float>(0, 2);

            AssertArray.AreEqual(b, a[Range(1, 3), 2]);
            AssertArray.AreEqual(b, a[Range(1, -1), -2]);
        }

        [TestMethod]
        public void TestAssign1D_in_2D()
        {
            var a = NN.Diag<float>(0, 1, 2, 3);

            a[Range(1, 3), 2] = new float[] { 0, 4 };
            AssertArray.AreEqual(NN.Diag<float>(0, 1, 4, 3), a);

            a[Range(1, -1), -2] = new float[] { 0, 5 };
            AssertArray.AreEqual(NN.Diag<float>(0, 1, 5, 3), a);
        }

        [TestMethod]
        public void TestSlice2D_in_2D()
        {
            var a = NN.Diag<float>(1, 2, 3, 4);
            var b = NN.Diag<float>(2, 3);
            var c = a[Range(1, 3), Range(1, 3)];
            AssertArray.AreEqual(b, c);
            c = a[Range(1, -1), Range(1, -1)];
            AssertArray.AreEqual(b, c);
        }

        [TestMethod]
        public void TestAssign2D_in_2D()
        {
            var a = NN.Diag<float>(0, 1, 2, 3);

            var b = NN.Array(new float[,]{
                { 5, 0 },
                { 0, 6 }
            });

            a[Range(1, 3), Range(1, 3)] = b;
            AssertArray.AreEqual(a, NN.Diag<float>(0, 5, 6, 3));
        }

        [TestMethod]
        public void TestSlice2D_in_3D()
        {
            var a = NN.Ones<float>(3, 4, 2);
            a.Item[1, 0, 0] = 3;
            var b = NN.Ones<float>(2, 3);
            b.Item[0, 0] = 3;
            AssertArray.AreEqual(b, a[From(1), Until(-1), 0]);
        }

        [TestMethod]
        public void TestSlice2D_in_3D_middle()
        {
            var a = NN.Ones<float>(3, 4, 2);
            a.Item[1, 1, 0] = 3;
            var b = NN.Ones<float>(2, 2);
            b.Item[0, 0] = 3;
            AssertArray.AreEqual(b, a[From(1), 1, _]);
        }

        [TestMethod]
        public void TestAssign2D_in_3D()
        {
            var a = NN.Ones<float>(3, 4, 2);

            var b = NN.Array(new float[,] {
                { 5, 2 },
                { 3, 6 }
            });

            a[Range(1, 3), Range(1, 3), 1] = b;
            Assert.AreEqual(5, a.Item[1, 1, 1]);
            Assert.AreEqual(2, a.Item[1, 2, 1]);
            Assert.AreEqual(3, a.Item[2, 1, 1]);
            Assert.AreEqual(6, a.Item[2, 2, 1]);

            a = NN.Ones<float>(3, 4, 2);
            a[Range(1, 3), 2, _] = b;
            Assert.AreEqual(5, a.Item[1, 2, 0]);
            Assert.AreEqual(2, a.Item[1, 2, 1]);
            Assert.AreEqual(3, a.Item[2, 2, 0]);
            Assert.AreEqual(6, a.Item[2, 2, 1]);

            a = NN.Ones<float>(3, 4, 2);
            a[2, Range(1, 3), _] = b;
            Assert.AreEqual(5, a.Item[2, 1, 0]);
            Assert.AreEqual(2, a.Item[2, 1, 1]);
            Assert.AreEqual(3, a.Item[2, 2, 0]);
            Assert.AreEqual(6, a.Item[2, 2, 1]);
        }

        [TestMethod]
        public void TestAssign1D_in_3D()
        {
            var b = NN.Array<float>(3, 5, 2);

            var a = NN.Ones<float>(5, 4, 3);
            a[Range(1, 4), 2, 1] = b;
            Assert.AreEqual(3, a.Item[1, 2, 1]);
            Assert.AreEqual(5, a.Item[2, 2, 1]);
            Assert.AreEqual(2, a.Item[3, 2, 1]);

            a = NN.Ones<float>(5, 4, 3);
            a[3, Range(1, 4), 1] = b;
            Assert.AreEqual(3, a.Item[3, 1, 1]);
            Assert.AreEqual(5, a.Item[3, 2, 1]);
            Assert.AreEqual(2, a.Item[3, 3, 1]);

            a = NN.Ones<float>(5, 4, 3);
            a[3, 2, _] = b;
            Assert.AreEqual(3, a.Item[3, 2, 0]);
            Assert.AreEqual(5, a.Item[3, 2, 1]);
            Assert.AreEqual(2, a.Item[3, 2, 2]);
        }

        [TestMethod]
        public void TestSetBroadCast()
        {
            float r = 0.5f;
            var M = NN.Random.Uniform(r, r, 40, 30).As<float>();
            var M2 = NN.Eye<float>(30).Scale(r);
            M[Until(30), _] = M2;
            AssertArray.AreEqual(M2, M[Until(30), _]);

            M2 = NN.Eye<float>(2).Scale(r);

            M = NN.Random.Uniform(-r, r, 5, 4, 3).As<float>();
            M[Range(2, 4), Range(1, 3), 1] = M2;
            AssertArray.AreEqual(M2, M[Range(2, 4), Range(1, 3), 1]);

            M = NN.Random.Uniform(-r, r, 5, 4, 3).As<float>();
            M[1, Range(1, 3), Range(1, 3)] = M2;
            AssertArray.AreEqual(M2, M[1, Range(1, 3), Range(1, 3)]);

            M = NN.Random.Uniform(-r, r, 5, 4, 3).As<float>();
            M[Range(2, 4), -1, Range(1, 3)] = M2;
            AssertArray.AreEqual(M2, M[Range(2, 4), -1, Range(1, 3)]);
        }

        [TestMethod]
        public void TestSetReshape()
        {
            float r = 0.5f;
            var M = NN.Random.Uniform(r, r, 40, 30).As<float>();
            var M2 = NN.Eye<float>(30).Scale(r);
            M[Until(30), _] = M2;
            AssertArray.AreEqual(M2, M[Until(30), _]);

            M2 = NN.Eye<float>(2).Scale(r);

            M = NN.Random.Uniform(-r, r, 5, 4, 3).As<float>();
            M[Range(2, 4), Range(1, 3), 1] = M2.Reshape(2, 2, 1);
            AssertArray.AreEqual(M2, M[Range(2, 4), Range(1, 3), 1]);

            M = NN.Random.Uniform(-r, r, 5, 4, 3).As<float>();
            M[1, Range(1, 3), Range(1, 3)] = M2.Reshape(1, 2, 2);
            AssertArray.AreEqual(M2, M[1, Range(1, 3), Range(1, 3)]);

            M = NN.Random.Uniform(-r, r, 5, 4, 3).As<float>();
            M[Range(2, 4), -1, Range(1, 3)] = M2.Reshape(2, 1, 2);
            AssertArray.AreEqual(M2, M[Range(2, 4), -1, Range(1, 3)]);
        }

        [TestMethod]
        public void TestDotRightWithSlice()
        {
            var x = NN.Array<float>(2, 3);

            var m = NN.Array<float>(new float[,] {
                { 0, -1, 2, 3 },
                { 1, -2, 3, 4 },
                { 2, -3, 4, 5 }
            });

            var m2 = m[From(1), Range(1, 3)];

            var m2Expected = NN.Array<float>(new float[,] {
                { -2, 3 },
                { -3, 4 }
            });
            AssertArray.AreAlmostEqual(m2Expected, m2);

            //Simple dot
            var result = m2Expected.Dot(x);
            var expected = NN.Array<float>(5, 6);
            AssertArray.AreAlmostEqual(expected, result);

            var result2 = m2.Dot(x);
            var expected2 = m2Expected.Dot(x);
            AssertArray.AreAlmostEqual(expected2, result2);
        }

        [TestMethod]
        public void TestSlice()
        {
            //var a = new Tensor(4, 3);
            var m = NN.Ones<float>(5, 5);
            var a = m[Range(1, 4), Range(1, 5)].T;
            var a2 = NN.Zeros<float>(4, 3);
            var b = new Array<float>(3, 5);

            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    a.Item[i, j] = i + j;
                    a2.Item[i, j] = i + j;
                }
            }
            AssertArray.AreAlmostEqual(a, a2);

            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 5; j++)
                    b.Item[i, j] = i - j;
            }

            var expected = NN.Array<float>(new float[,] {
                { 1, 1, 1, 1, 1 },
                { 1, 0, 1, 2, 3 },
                { 1, 1, 2, 3, 4 },
                { 1, 2, 3, 4, 5 },
                { 1, 1, 1, 1, 1 }
            });
            AssertArray.AreAlmostEqual(m, expected);

            AssertArray.AreAlmostEqual(a.Dot(b), a2.Dot(b));
        }

        [TestMethod, ExpectedException(typeof(RankException)), TestCategory("Exception")]
        public void AssignWrongSliceThrowsException()
        {
            var x = NN.Zeros(5, 3);
            x[1] = NN.Ones(4);
        }

        [TestMethod]
        public void TestStep()
        {
            var x = NN.Array(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);

            AssertArray.AreEqual(NN.Array( 1, 3, 5 ), x[Range(1, 7, 2)]);
            AssertArray.AreEqual(NN.Array( 8, 9 ), x[Range(-2, 10)]);
            AssertArray.AreEqual(NN.Array( 7, 6, 5, 4 ), x[Range(-3, 3, -1)]);
            AssertArray.AreEqual(NN.Array( 5, 6, 7, 8, 9 ), x[From(5)]);
            AssertArray.AreEqual(NN.Array( 5, 7, 9 ), x[From(5, step: 2)]);
            AssertArray.AreEqual(NN.Array( 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 ), x[Step(-1)]);

            var y = NN.Array(new [,] { { 0, 10 }, { 1, 11 } });
            AssertArray.AreEqual(NN.Array( 1, 11 ), y[Range(-1, 0, -1)], allowBroadcasting: true);
            AssertArray.AreEqual(NN.Array( 0, 10 ), y[Range(-2, null, -1)], allowBroadcasting: true);

            AssertArray.AreEqual(NN.Empty<int>(), y[Range(-2, 0, -1)], allowBroadcasting: true);
        }

        [TestMethod, ExpectedException(typeof(ArgumentException)), TestCategory("Exception")]
        public void FailStep()
        {
            var y = NN.Array(new float[,] { { 0, 10 }, { 1, 11 } });
            AssertArray.AreAlmostEqual(NN.Empty<float>(), y[Range(-3, null, -1)]);
        }

        [TestMethod]
        public void TestAdvancedIndexing()
        {
            var x = NN.Range<int>(10, 1, -1);
            AssertArray.AreEqual(NN.Array( 10, 9, 8, 7, 6, 5, 4, 3, 2 ), x);

            AssertArray.AreEqual(new [] { 7, 7, 9, 2 }, x[NN.Array(3, 3, 1, 8)]);
            AssertArray.AreEqual(new [] { 7, 7, 4, 2 }, x[NN.Array(3, 3, -3, 8 )]);
            var test = x[NN.Array(new[,] { { 1, 1 }, { 2, 3 } })];
            AssertArray.AreEqual(new [,] { { 9, 9 }, { 8, 7 } }, test);

#if ZERO
            var y = NN.Range<float>(35).Reshape(5, 7);
            AssertArray.AreEqual(NN.Array( 0, 15, 30 }, y[new[] { 0, 2, 4 }, new[] { 0, 1, 2 }]);

            // The broadcasting mechanism permits index arrays to be combined with scalars for other indices. The effect is that the scalar
            // value is used for all the corresponding values of the index arrays:

            AssertArray.AreEqual(NN.Array( 1, 15, 29 }, y[new[] { 0, 2, 4 }, 1]);
            // >>>>>> y[np.array([0,2,4]), 1]
            // array([ 1, 15, 29])


            // Jumping to the next level of complexity, it is possible to only partially index an array with index arrays. It takes a bit of
            // thought to understand what happens in such cases. For example if we just use one index array with y:

            AssertArray.AreEqual(new float[,] {
                { 0, 1, 2, 3, 4, 5, 6 },
                { 14, 15, 16, 17, 18, 19, 20 },
                { 28, 20, 30, 31, 32, 33, 34 }
            }, y[new[] { 0, 2, 4 }]);
            // >>>>>> y[np.array([0,2,4])]
            // array([[ 0,  1,  2,  3,  4,  5,  6],
            //        [14, 15, 16, 17, 18, 19, 20],
            //        [28, 29, 30, 31, 32, 33, 34]])


            //What results is the construction of a new array where each value of the index array selects one row from the array being indexed
            //and the resultant array has the resulting shape (size of row, number index elements).
#endif
        }

        [TestMethod]
        public void TestAdvancedIndexing2()
        {
            var emb = NN.Range(3000 * 5).Reshape(3000, 5);
            var test = emb[NN.Array(new[,] { { 3, 4 }, { 3, 2 } })];
            var expected = NN.Array(new[,,] {
                { { 15, 16, 17, 18, 19 },
                { 20, 21, 22, 23, 24 } },

                { { 15, 16, 17, 18, 19 },
                { 10, 11, 12, 13, 14} } });
            AssertArray.AreEqual(expected, test);
        }

        [TestMethod]
        public void TestAdvancedIndexing3()
        {
            var emb = NN.Range(3000 * 5).Reshape(3000, 5);
            var test = emb[new[] { 3, 4 }, new[] { 3, 2 }];
            var expected = NN.Array( 18, 22 );
            AssertArray.AreEqual(expected, test);
        }

        [TestMethod, ExpectedException(typeof(IndexOutOfRangeException)), TestCategory("Exception")]
        public void FailIndexing1()
        {
            var x = NN.Range<float>(10, 1, -1);
            var bad = x[NN.Array(3, 3, 20, 8)];    // index 20 out of bounds 0<=index<9
        }

        //[TestMethod, ExpectedException(typeof(Exception)), TestCategory("Exception")]
        //public void FailIndexing2()
        //{
        //    var y = NN.Range<float>(35).Reshape(5, 7);
        //    var bad = y[new[] { 0, 2, 4 }, new[] { 0, 1 }];   // shape mismatch: objects cannot be broadcast to a single shape
        //}

        [TestMethod]
        public void CanSliceArrayOfOneElement()
        {
            var a = NN.Ones(1);
            var b = NN.Ones(1);
            AssertArray.AreEqual(b, a[Step(-1)]);
            AssertArray.AreEqual(b, a[_]);
        }

        [TestMethod]
        public void CanSetWholeArray()
        {
            var a = NN.Diag<float>(0, 1, 2, 3);
            var b = NN.Ones<float>(4, 4);

            a._ = b;
            AssertArray.AreEqual(b, a);
            Assert.AreNotEqual(b, a);
        }
    }
}
