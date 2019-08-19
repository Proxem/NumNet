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
using System.Linq;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Proxem.NumNet.Single;

namespace Proxem.NumNet.Test
{
    [TestClass]
    public class UnitTestIterable
    {
        [TestMethod]
        public void TestIterateOnMatrix()
        {
            var m1 = NN.Array(new float[,] {
                { 0, 1, 2, 3 },
                { 1, 2, 3, 4 },
                { 2, 3, 4, 5 }
            });

            var s1 = string.Join("", m1);
            Assert.AreEqual("012312342345", s1);

            var x = new List<int> { 0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5 };
            var eq = x.Zip(m1, Tuple.Create).Min(xy => xy.Item1 == xy.Item2);
            Assert.IsTrue(eq);

            var m2 = m1[1.., 1..3];
            var s2 = string.Join("", m2);
            Assert.AreEqual("2334", string.Join("", m2));

            x = new List<int>() { 2, 3, 3, 4 };
            eq = x.Zip(m2, Tuple.Create).Min(xy => xy.Item1 == xy.Item2);
            Assert.IsTrue(eq);

            var m3 = m1[1, 1..3];
            x = new List<int>() { 2, 3 };
            eq = x.Zip(m2, Tuple.Create).Min(xy => xy.Item1 == xy.Item2);
            Assert.IsTrue(eq);
        }

        [TestMethod]
        public void TestIterateOnMatrix2()
        {
            var m1 = NN.Range<float>(3 * 4 * 5).Reshape(3, 4, 5);
            Assert.AreEqual(m1.Size, m1.Count());

            var s1 = string.Join(",", m1) + ",";
            var expected1 = new StringBuilder();
            var l1 = new List<float>();
            m1.Apply((f) => { expected1.Append(f); expected1.Append(','); l1.Add(f); return f; });
            Assert.AreEqual(l1.Count, m1.Size);
            Assert.AreEqual(expected1.ToString(), s1);

            var m2 = m1[1..3, 2..4, 2..5];
            Assert.AreEqual(m2.Size, m2.Count());

            var s2 = string.Join(",", m2) + ",";
            var expected2 = new StringBuilder();
            var l2 = new List<float>();
            m2.Apply((f) => { expected2.Append(f); expected2.Append(','); l2.Add(f); return f; });
            Assert.AreEqual(l2.Count, m2.Size);

            var expected3 = new StringBuilder();
            for (var i = 0; i < m2.Shape[0]; i++)
                for (var j = 0; j < m2.Shape[1]; j++)
                    for (var k = 0; k < m2.Shape[2]; k++)
                    {
                        expected3.Append(m2.Item[i, j, k]);
                        expected3.Append(',');
                    }
            Assert.AreEqual(expected3.ToString(), expected2.ToString());
            Assert.AreEqual(expected2.ToString(), s2);
        }

        [TestMethod]
        public void TestGetOffsets()
        {
            // offsets of vec
            var v = NN.Range<float>(5);
            var offsets = v.GetOffsets(new[] { 5 }).ToArray();
            AssertArray.AreEqual(new[] { 0 }, offsets);

            // offsets of mat
            var m = NN.Range<float>(3 * 4).Reshape(3, 4);
            offsets = m.GetOffsets(new[] { 3, 4 }).ToArray();
            AssertArray.AreEqual(new[] { 0, 4, 8 }, offsets);

            // offsets of sliced mat
            offsets = m[.., 1..3].GetOffsets(new[] { 3, 2 }).ToArray();
            AssertArray.AreEqual(new[] { 1, 5, 9 }, offsets);

            // offsets of sliced vec
            offsets = v[3..].GetOffsets(new[] { 5 }).ToArray();
            AssertArray.AreEqual(new[] { 3 }, offsets);

            // offsets of left broadcasted vec
            offsets = v.Reshape(1, 5).GetOffsets(new[] { 4, 5 }).ToArray();
            AssertArray.AreEqual(new[] { 0, 0, 0, 0 }, offsets);
        }
    }
}