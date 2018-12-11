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
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Proxem.NumNet.Test
{
    using static Slicer;

    [TestClass]
    public class TestFlags
    {
        [TestMethod]
        public void TestContiguousOnVec()
        {
            var a = NN.Array<float>( 0f, 1, 2, 3 );

            a.AssertIsContiguous();
            a[Range(1, 3)].AssertIsContiguous();
            a[Upto(3)].AssertIsContiguous();
            a[From(2)].AssertIsContiguous();
            a[Only(2)].AssertIsContiguous();

            a[Step(-1)].AssertIsContiguous();
            a[Step(2)].AssertIsContiguous();
        }

        [TestMethod]
        public void TestContiguousOnMat()
        {
            var a = NN.Array(
                new float[] { 0, 1, 2, 3 },
                new float[] { 1, 2, 3, 4 },
                new float[] { 2, 3, 4, 5 }
            );

            a.AssertIsContiguous();
            a[Range(1, 3)].AssertIsContiguous();
            a[Upto(3)].AssertIsContiguous();
            a[From(2)].AssertIsContiguous();

            a[Range(1, 3), Upto(3)].AssertIsNotContiguous();
            a[Upto(3), From(2)].AssertIsNotContiguous();
            a[From(2), Range(1, 3)].AssertIsNotContiguous();
            a[Only(2), Range(1, 3)].AssertIsContiguous();

            a[From(2), Only(1)].AssertIsContiguous();
            a[Range(1, 3), Only(2)].AssertIsContiguous();

            a[Step(-1)].AssertIsNotContiguous();
            a[Step(2)].AssertIsNotContiguous();
        }

        [TestMethod]
        public void TestTranspose()
        {
            var t = NN.Ones<float>(5, 4, 3);
            t.AssertOfShape(5, 4, 3);

            t.Transpose(0, 1, 2).AssertIsNotTransposed();
            t.Transpose(0, 2, 1).AssertIsTransposed();

            t.Transpose(1, 0, 2).AssertIsTransposed();
            t.Transpose(1, 2, 0).AssertIsTransposed();

            t.Transpose(2, 0, 1).AssertIsTransposed();
            t.Transpose(2, 1, 0).AssertIsTransposed();

            t.Transpose(0, 2, 1).Transpose(0, 2, 1).AssertIsNotTransposed();
            t.Transpose(2, 0, 1).Transpose(1, 2, 0).AssertIsNotTransposed();
        }
    }
}
