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
    public class TestBroadCast
    {
        [TestMethod]
        public void CanBroadcastScalarToArray()
        {
            // http://www.onlamp.com/pub/a/python/2000/09/27/numerically.html
            /*

            a = np.array([[1,2],[3, 4]])
            b = 1
            a + b

            */
            var a = NN.Array(new[,] { { 1, 2 }, { 3, 4 } });
            var b = 1;
            var r = NN.Array(new[,] { { 2, 3 }, { 4, 5 } });

            AssertArray.GenerateTests(a, NN.Ones<int>, a1 => AssertArray.AreEqual(r, a1 + b));
        }

        [TestMethod, ExpectedException(typeof(RankException)), TestCategory("Exception")]
        public void CantBroadcast_2_3_to_3_2()
        {
            // http://www.onlamp.com/pub/a/python/2000/09/27/numerically.html?page=2
            /*

            a = np.array([[1, 2, 3], [4, 5, 6]])
            b = np.array([[1, 2], [3, 4], [5, 6]])
            a + b

            */
            var a = NN.Array(new[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var b = NN.Array(new[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
            var c = a + b;
        }

        [TestMethod]
        public void CanBroadcast_3_to_2_3()
        {
            // http://www.onlamp.com/pub/a/python/2000/09/27/numerically.html?page=2
            /*

a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[7, 8, 9]])
a + b

            */
            var a = NN.Array(new[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var c = NN.Array(new[] { 7, 8, 9 });
            var rAdd = NN.Array(new[,] { { 8, 10, 12 }, { 11, 13, 15 } });
            var rMul = NN.Array(new[,] { { 7, 16, 27 }, { 28, 40, 54 } });

            a.AssertOfShape(2, 3);
            c.AssertOfShape(3);

            AssertArray.GenerateTests(a, c, NN.Ones<int>, (a1, c1) => AssertArray.AreEqual(rAdd, a1 + c1));
            AssertArray.GenerateTests(a, c, NN.Zeros<int>, (a1, c1) => AssertArray.AreEqual(rMul, a1 * c1));
        }

        [TestMethod]
        public void CanBroadcast_1_to_5()
        {
            // http://www.onlamp.com/pub/a/python/2000/09/27/numerically.html?page=2
            /*

x=np.zeros((3,4,5,6,7))
y=np.zeros((7,))
z=x+y

            */
            var x = NN.Zeros(3, 4, 5, 6, 7);
            var y = NN.Zeros(7);
            var z = x + y;  // no exception
        }

        [TestMethod]
        public void CanBroadcast_2_to_3_1()
        {
            // http://www.onlamp.com/pub/a/python/2000/09/27/numerically.html?page=2
            /*

z=np.array([1, 2])
v=np.array([[3], [4], [5]])
z+v

            */
            // When comparing the size of each axis, if either one of the compared axes has a size of one, broadcasting can also occur
            var z = NN.Array(new[] { 1, 2 });
            AssertArray.AreEqual(z.Shape, new int[] { 2 });

            var v = NN.Array(new[,] { { 3 }, { 4 }, { 5 } });
            AssertArray.AreEqual(v.Shape, new int[] { 3, 1 });

            AssertArray.AreEqual(new[,] { { 4, 5 }, { 5, 6 }, { 6, 7 } }, z + v);
            //In this form, the first multiarray z was extended to a (3,2) multiarray and the second multiarray v was extended to a (3,2) multiarray.
            //Essentially, broadcasting occurred on both operands! This only occurs when the axis size of one of the multiarrays has the value of one.
        }

        [TestMethod]
        public void CanBroadcast_2_to_3_1_bis()
        {
            // http://www.onlamp.com/pub/a/python/2000/09/27/numerically.html?page=3
            /*

z = np.array([1,2])
w = np.array([3,4,5])
z+np.reshape(w,(3,1))

a=np.zeros((3,4,5,6))
b=np.zeros((4,6))
c=a+b[:,np.newaxis,:]

            */

            var z = NN.Array(new[] { 1, 2 });
            var w = NN.Array(new[] { 3, 4, 5 });

            AssertArray.AreEqual(new[,] { { 4, 5 }, { 5, 6 }, { 6, 7 } }, z + w.Reshape(3, 1));
        }

        [TestMethod]
        public void CanInsertNewAxisAndBroadcast()
        {
            // http://www.onlamp.com/pub/a/python/2000/09/27/numerically.html?page=3
            /*

a=np.zeros((3,4,5,6))
b=np.zeros((4,6))
c=a+b[:,np.newaxis,:]

            */

            var a = NN.Zeros(new[] { 3, 4, 5, 6 });
            var b = NN.Zeros(new[] { 4, 6 });
            var c = a + b[.., NewAxis, ..];

            AssertArray.AreEqual(c.Shape, new[] { 3, 4, 5, 6 });
        }

        [TestMethod]
        public void CanInsertNewAxis()
        {
            // http://www.onlamp.com/pub/a/python/2000/09/27/numerically.html?page=3
            /*

b=np.arange(2 * 3).reshape(2, 3)
b[:,np.newaxis,:]

            */

            var b = NN.Range(2 * 3).Reshape(2, 3);
            var c = b[.., NewAxis, ..];

            AssertArray.AreEqual(c.Shape, new[] { 2, 1, 3 });
            AssertArray.AreEqual(c.Stride, new[] { 3, 0, 1 });
        }
    }
}