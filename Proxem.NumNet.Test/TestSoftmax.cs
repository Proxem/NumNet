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
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Proxem.NumNet;


namespace Proxem.Theanet.Test
{
    [TestClass]
    public class TestSoftmax
    {
        //Array<float> softmax(Array<float> w)
        //{
        //    w = NN.Array(w);    // lift Array<Array<T>> => Array<T>
        //    var maxes = NN.Max(w, axis: w.Shape.Length - 1, keepDims: true);
        //    var e = NN.Exp(w - maxes);
        //    var sum = NN.Sum(e, axis: w.Shape.Length - 1, keepDims: true);
        //    var result = e / sum;
        //    // TODO: result.Shape.Length == 2
        //    //if (result.Shape.Length == 1) result = result[Slicer.NewAxis, Slicer._];
        //    return result;
        //}

        [TestMethod]
        public void TestSoftmax2D()
        {
            var X = NN.Array(new float[,] { { 1, 3 }, { 2, 5 } });

            AssertArray.AreAlmostEqual(new float[,] { { 0.11920292f, 0.88079708f }, { 0.04742587f, 0.95257413f } }, NN.Softmax(X));
        }

        [TestMethod]
        public void TestSoftmax1D()
        {
            var X = NN.Array(new float[] { 1, 3 });

            AssertArray.AreAlmostEqual(new float[] { 0.11920292f, 0.88079708f }, NN.Softmax(X));
        }

        [TestMethod]
        public void TestLogSumExp()
        {
            var X = NN.Array(new float[,] { { 1, 3 }, { 2, 5 } });

            AssertArray.AreAlmostEqual(NN.Log(NN.Sum(NN.Exp(X), axis: -1)), NN.LogSumExp(X));
        }

    }
}
