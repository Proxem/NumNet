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
using Proxem.BlasNet;

namespace Proxem.NumNet.Test
{
    [TestClass]
    public class TestSparseMatrix
    {
        IBlas Provider = Blas.Provider;

        [TestMethod]
        public void TestSparseMatrixMultiplication()
        {
            // https://en.wikipedia.org/wiki/Sparse_matrix
            var aDense = new float[]
            {
                0, 0, 0, 0,
                5, 8, 0, 0,
                0, 0, 3, 0,
                0, 6, 0, 0
            };

            var a  = new float[] { 5, 8, 3, 6 };
            var ja = new int[] { 0, 1, 2, 1 };
            var ia = new int[] { 0, 0, 2, 3, 4 };

            var bDense = new float[]
            {
                1, 2, 0, 0, 0, 0,
                0, 3, 0, 4, 0, 0,
                0, 0, 5, 6, 7, 0,
                0, 0, 0, 0, 0, 8,
            };

            var cDense = new float[4 * 6];
            var c = new float[4 * 6];

            Provider.sgemm(Order.RowMajor, Transpose.NoTrans, Transpose.NoTrans, 4, 6, 4, 1, aDense, 0, 4, bDense, 0, 6, 0, cDense, 0, 6);
            Provider.scsrmm(Transpose.NoTrans, 4, 6, 4, 1, a, 0, ja, 0, ia, 0, bDense, 0, 6, 0, c, 0, 6);

            AssertArray.AreEqual(cDense, c);
        }
    }
}
