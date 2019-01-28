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
    [TestClass]
    public class TestShuffle
    {
        [TestMethod]
        public void TestShuffleDim1()
        {
            var a = NN.Range(10);
            int[] perm = new int[10] { 1, 2, 3, 8, 9, 0, 4, 6, 7, 5 };
            var expected = NN.Array(new int[10] { 5, 0, 1, 2, 6, 9, 7, 8, 3, 4 });
            var shufA = a.Shuffle(perm);
            AssertArray.AreEqual(shufA, expected);
        }

        [TestMethod]
        public void TestShuffleInPlaceDim1()
        {
            var a = NN.Range(10);
            int[] perm = new int[10] { 1, 2, 3, 8, 9, 0, 4, 6, 7, 5 };
            var expected = NN.Array(new int[10] { 5, 0, 1, 2, 6, 9, 7, 8, 3, 4 });
            a.ShuffleInplace(perm);
            AssertArray.AreEqual(a, expected);
        }

        [TestMethod]
        public void TestShuffleDim2()
        {
            var a = NN.Range(20).Reshape(4, 5);

            int[] permsRow = new int[4] { 2, 0, 3, 1 };
            int[] permsCol= new int[5] { 0, 3, 1, 4, 2 };
            var expectedRow = NN.Array(new int[20] {5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 0, 1, 2, 3, 4, 10, 11, 12, 13, 14 }).Reshape(4, 5);
            var expectedCol = NN.Array(new int[20] {0, 2, 4, 1, 3, 5, 7, 9, 6, 8, 10, 12, 14, 11, 13, 15, 17, 19, 16, 18 }).Reshape(4, 5);

            var shufARow = a.Shuffle(perms: permsRow);
            var shufACol = a.Shuffle(perms: permsCol, axis: 1);

            AssertArray.AreEqual(shufARow, expectedRow);
            AssertArray.AreEqual(shufACol, expectedCol);
        }


        [TestMethod]
        public void TestShuffleInplaceDim2()
        {
            var a = NN.Range(20).Reshape(4, 5);
            var b = NN.Range(20).Reshape(4, 5);

            int[] permsRow = new int[4] { 2, 0, 3, 1 };
            int[] permsCol = new int[5] { 0, 3, 1, 4, 2 };
            var expectedRow = NN.Array(new int[20] { 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 0, 1, 2, 3, 4, 10, 11, 12, 13, 14 }).Reshape(4, 5);
            var expectedCol = NN.Array(new int[20] { 0, 2, 4, 1, 3, 5, 7, 9, 6, 8, 10, 12, 14, 11, 13, 15, 17, 19, 16, 18 }).Reshape(4, 5);

            a.ShuffleInplace(perms: permsRow);
            b.ShuffleInplace(perms: permsCol, axis: 1);

            AssertArray.AreEqual(a, expectedRow);
            AssertArray.AreEqual(b, expectedCol);
        }

        [TestMethod]
        public void TestShuffleDim3()
        {
            var a = NN.Range(24).Reshape(3, 2, 4);

            int[] perms1 = new int[3] { 0, 2, 1 };
            int[] perms2 = new int[2] { 1, 0 };
            int[] perms3 = new int[4] { 3, 1, 0, 2 };
            var expected1 = NN.Array(new int[24] { 0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 8, 9, 10, 11, 12, 13, 14, 15 }).Reshape(3, 2, 4);
            var expected2 = NN.Array(new int[24] { 4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11, 20, 21, 22, 23, 16, 17, 18, 19 }).Reshape(3, 2, 4);
            var expected3 = NN.Array(new int[24] { 2, 1, 3, 0, 6, 5, 7, 4, 10, 9, 11, 8, 14, 13, 15, 12, 18, 17, 19, 16, 22, 21, 23, 20 }).Reshape(3, 2, 4);

            var shufA1 = a.Shuffle(perms: perms1);
            var shufA2 = a.Shuffle(perms: perms2, axis: 1);
            var shufA3 = a.Shuffle(perms: perms3, axis: 2);

            AssertArray.AreEqual(shufA1, expected1);
            AssertArray.AreEqual(shufA2, expected2);
            AssertArray.AreEqual(shufA3, expected3);
        }

        [TestMethod]
        public void TestShuffleInplaceDim3()
        {
            var a = NN.Range(24).Reshape(3, 2, 4);
            var b = NN.Range(24).Reshape(3, 2, 4);
            var c = NN.Range(24).Reshape(3, 2, 4);

            int[] perms1 = new int[3] { 0, 2, 1 };
            int[] perms2 = new int[2] { 1, 0 };
            int[] perms3 = new int[4] { 3, 1, 0, 2 };
            var expected1 = NN.Array(new int[24] { 0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 8, 9, 10, 11, 12, 13, 14, 15 }).Reshape(3, 2, 4);
            var expected2 = NN.Array(new int[24] { 4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11, 20, 21, 22, 23, 16, 17, 18, 19 }).Reshape(3, 2, 4);
            var expected3 = NN.Array(new int[24] { 2, 1, 3, 0, 6, 5, 7, 4, 10, 9, 11, 8, 14, 13, 15, 12, 18, 17, 19, 16, 22, 21, 23, 20 }).Reshape(3, 2, 4);

            a.ShuffleInplace(perms: perms1);
            b.ShuffleInplace(perms: perms2, axis: 1);
            c.ShuffleInplace(perms: perms3, axis: 2);

            AssertArray.AreEqual(a, expected1);
            AssertArray.AreEqual(b, expected2);
            AssertArray.AreEqual(c, expected3);
        }
    }
}
