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
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Proxem.NumNet.Single;

namespace Proxem.NumNet.Test
{
    using static EinsteinSumTools;

    [TestClass]
    public class TestEinsteinSum
    {
        [TestMethod]
        public void CanReadEinsteinStringForInner()
        {
            var einstein = EinsteinRead("i,i->");
            Assert.AreEqual(1, einstein.Length);
            Assert.AreEqual(Inner(0, 0), einstein[0]);
        }

        [TestMethod]
        public void CanReadEinsteinStringForOuter()
        {
            var einstein = EinsteinRead("i,j->ij");
            Assert.AreEqual(2, einstein.Length);
            Assert.AreEqual(OuterX(0, 0), einstein[0]);
            Assert.AreEqual(OuterY(0, 1), einstein[1]);

            einstein = EinsteinRead("i,j->ji");
            Assert.AreEqual(2, einstein.Length);
            Assert.AreEqual(OuterX(0, 1), einstein[0]);
            Assert.AreEqual(OuterY(0, 0), einstein[1]);
        }

        [TestMethod]
        public void CanReadEinsteinStringForSum()
        {
            var einstein = EinsteinRead("i,->");
            Assert.AreEqual(1, einstein.Length);
            Assert.AreEqual(SumX(0), einstein[0]);

            einstein = EinsteinRead(",i->");
            Assert.AreEqual(1, einstein.Length);
            Assert.AreEqual(SumY(0), einstein[0]);
        }


        [TestMethod]
        public void EinsteinShapeForInnerIsCorrect()
        {
            var einstein = EinsteinRead("i,i->");
            var shape = EinsteinShape(new[] { 10 }, new[] { 10 }, einstein);
            Assert.AreEqual(0, shape.Length);
        }

        [TestMethod]
        public void EinsteinShapeForOuterIsCorrect()
        {
            var einstein = EinsteinRead("i,j->ij");
            var shape = EinsteinShape(new[] { 10 }, new[] { 15 }, einstein);
            Assert.AreEqual(2, shape.Length);
            Assert.AreEqual(10, shape[0]);
            Assert.AreEqual(15, shape[1]);
        }

        [TestMethod]
        public void InnerAsEinsteinSumMatchesInner()
        {
            var x = NN.Random.Uniform(-1f, 1f, 10);
            var y = NN.Random.Uniform(-1f, 1f, 10);
            AssertArray.AreEqual(x.Dot(y), NN.EinsteinSum(x, y, "i,i->"));
        }

        [TestMethod]
        public void OuterAsEinsteinSumMatchesOuter()
        {
            var x = NN.Random.Uniform(-1f, 1f, 10);
            var y = NN.Random.Uniform(-1f, 1f, 15);
            AssertArray.AreEqual(x.Outer(y), NN.EinsteinSum(x, y, "i,j->ij"));
        }

        [TestMethod]
        public void SumAsEinsteinSumMatchesSum()
        {
            var x = NN.Random.Uniform(-1f, 1f, 10);
            var y = NN.Ones(1);
            AssertArray.AreEqual(x.Sum(axis: 0), NN.EinsteinSum(x, y, "i,->"));
        }

        [TestMethod]
        public void DotAsEinsteinSumMatchesDot()
        {
            var x = NN.Random.Uniform(-1f, 1f, 10, 20);
            var y = NN.Random.Uniform(-1f, 1f, 20, 13);
            AssertArray.AreAlmostEqual(x.Dot(y), NN.EinsteinSum(x, y, "ij,jk->ik"));
        }

        [TestMethod]
        public void TensorDotAsEinsteinSumMatchesTensorDot()
        {
            var x = NN.Random.Uniform(-1f, 1f, 10, 20);
            var y = NN.Random.Uniform(-1f, 1f, 20, 13, 5);
            AssertArray.AreAlmostEqual(NN.TensorDot(x, new[] { 1 }, y, new[] { 0 }), NN.EinsteinSum(x, y, "ij,jkl->ikl"));
        }
    }
}
