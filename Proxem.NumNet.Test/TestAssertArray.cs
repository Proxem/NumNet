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
    [TestClass]
    public class TestAssertArray
    {
        [TestMethod]
        public void AssertOfDimAcceptCorrects()
        {
            NN.Empty<float>().AssertOfDim(0);
            NN.Zeros(1).AssertOfDim(1);
            NN.Zeros(1, 1).AssertOfDim(2);
            NN.Zeros(1, 1, 1).AssertOfDim(3);
        }

        [TestMethod]
        public void AssertOfDimThrowsException()
        {
            var a = NN.Zeros(1);
            var b = NN.Zeros(1, 1);
            var c = NN.Zeros(1, 1, 1);

            AssertThrows<RankException>(() => a.AssertOfDim(2));
            AssertThrows<RankException>(() => a.AssertOfDim(3));
            AssertThrows<RankException>(() => b.AssertOfDim(1));
            AssertThrows<RankException>(() => b.AssertOfDim(3));
            AssertThrows<RankException>(() => c.AssertOfDim(1));
            AssertThrows<RankException>(() => c.AssertOfDim(2));
        }

        [TestMethod]
        public void AssertOfShapeAcceptCorrects()
        {
            NN.Zeros(5).AssertOfShape(5);
            NN.Zeros(7).AssertOfShape(7);
            NN.Zeros(5, 7).AssertOfShape(5, 7);
            NN.Zeros(5, 3).AssertOfShape(5, 3);
            NN.Zeros(5, 3, 7).AssertOfShape(5, 3, 7);
            NN.Zeros(5, 3, 9).AssertOfShape(5, 3, 9);
        }

        [TestMethod]
        public void AssertOfShapeThrowsException()
        {
            AssertThrows<RankException>(() => NN.Zeros(5).AssertOfShape(7));
            AssertThrows<RankException>(() => NN.Zeros(5).AssertOfShape(5, 1));
            AssertThrows<RankException>(() => NN.Zeros(5, 7).AssertOfShape(5, 3));
            AssertThrows<RankException>(() => NN.Zeros(5, 7).AssertOfShape(1, 5, 7));
            AssertThrows<RankException>(() => NN.Zeros(5, 3, 7).AssertOfShape(5, 9, 7));
            AssertThrows<RankException>(() => NN.Zeros(5, 3, 7).AssertOfShape(1, 5, 3, 9));
        }

        [TestMethod]
        public void AssertAreEqualWorks()
        {
            var a = NN.Random.Uniform(-1f, 1f, 4, 3);
            var b = a.Copy();
            var c = a.T;
            AssertArray.AreEqual(a, b);
            AssertArray.AreNotEqual(a, c);
            Assert.IsTrue(a == a.T.T);
        }

        [TestMethod]
        public void AssertAreEqualWithBroadcastingWorks()
        {
            var a = NN.Random.Uniform(-1f, 1f, 5);
            var b = a.Reshape(1, 5);
            var c = a.Reshape(5, 1);
            AssertArray.AreEqual(a, b, allowBroadcasting: true);
            AssertArray.AreEqual(a, c, allowBroadcasting: true);
            AssertArray.AreEqual(c, b, allowBroadcasting: true);
            AssertArray.AreEqual(b, c, allowBroadcasting: true);

            AssertArray.AreNotEqual(b, c, allowBroadcasting: false);
            AssertArray.AreNotEqual(a, b, allowBroadcasting: false);
            AssertArray.AreNotEqual(a, c, allowBroadcasting: false);
        }

        public static T AssertThrows<T>(Action f) where T : Exception
        {
            try
            {
                f();
                throw new Exception("Exception not throwed");
            }
            catch(T e)
            {
                return e;
            }
        }
    }
}
