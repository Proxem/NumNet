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
using Proxem.NumNet.Double;

namespace Proxem.NumNet.Test
{
    [TestClass]
    public class TestLapack
    {
        [TestMethod]
        public void TestDeterminantUpper()
        {
            const int n = 4;
            float[] A = new float[n * n] { 1, 2, 4, 7,
                                           0, 3, 5, 8,
                                           0, 0, 6, 9,
                                           0, 0, 0, 10 };

            var result = Lapack.Determinant(A, n);
            Assert.AreEqual(result, 180f);
        }

        [TestMethod]
        public void TestDeterminant2x2()
        {
            const int n = 2;
            float[] A = new float[n * n] { 1, 2,
                                           3, 4 };
            var result = Lapack.Determinant(A, n);
            AssertArray.AreAlmostEqual(result, -2f);
        }


        [TestMethod]
        public void TestDeterminant2x2b()
        {
            const int n = 2;
            var A = new float[n, n] { { 1, 2 },
                                      { 3, 4 } };
            var result = Lapack.Determinant(A, n);
            AssertArray.AreAlmostEqual(result, -2f);
        }

        [TestMethod]
        public void TestDeterminant2x2c()
        {
            const int n = 2;
            var A = new float[n * n] { 1, 2 ,
                                       3, 4 };
            var result = Lapack.Determinant(A, n);
            AssertArray.AreAlmostEqual(result, -2f);
        }

        [TestMethod]
        public void TestInverse2x2b()
        {
            const int n = 2;
            float[,] a = new float[n, n] { { 1, 2 },
                                           { 3, 4 } };
            float[,] aInv = (float[,])a.Clone();
            Lapack.Inverse(aInv, n);

            float[,] eye = new float[n, n];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    for (int k = 0; k < n; k++)
                    {
                        eye[i, j] += a[i, k] * aInv[k, j];
                    }
                }
            }
            AssertArray.AreAlmostEqual(1f, eye[0, 0]);
            AssertArray.AreAlmostEqual(0f, eye[0, 1]);
            AssertArray.AreAlmostEqual(0f, eye[1, 0]);
            AssertArray.AreAlmostEqual(1f, eye[1, 1]);
        }

        [TestMethod]
        public void TestSolve()
        {
            /* Solve the equations A*X = B */
            // https://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/dgesv_ex.c.htm
            const int N = 5;
            const int NRHS = 3;
            const int LDA = N;
            const int LDB = NRHS;
            int n = N, nrhs = NRHS, lda = LDA, ldb = LDB;

            /* Local arrays */
            int[] ipiv = new int[N];
            double[] a = new double[N * N] {
                6.80, -6.05, -0.45,  8.32, -9.67,
               -2.11, -3.30,  2.58,  2.71, -5.14,
                5.66,  5.36, -2.70,  4.35, -7.26,
                5.97, -4.44,  0.27, -7.17,  6.08,
                8.23,  1.08,  9.04,  2.14, -6.87
            };
            double[] b = new double[N * NRHS] {
                4.02, -1.56,  9.81,
                6.19,  4.00, -4.09,
               -8.22, -8.67, -4.57,
               -7.57,  1.75, -8.61,
               -3.03,  2.86,  8.99
            };
            /* Solve the equations A*X = B */
            Lapack.gesv(n, nrhs, a, lda, ipiv, b, ldb);

            // Solution
            var solution = NN.Array(new[] {
              -0.80,  -0.39,   0.96,
              -0.70,  -0.55,   0.22,
               0.59,  0.84,   1.90,
               1.32,  -0.10,   5.36,
               0.57,   0.11,   4.04,
            }).Reshape(n, nrhs);
            AssertArray.AreAlmostEqual(solution, NN.Array(b).Reshape(N, NRHS), 1e-2, 1e-2);

            // Details of LU factorization
            var luFactorization = NN.Array(new[]
            {
               8.23,   1.08,   9.04,   2.14,  -6.87,
               0.83,  -6.94,  -7.92,   6.55,  -3.99,
               0.69,  -0.67, -14.18,   7.24,  -5.19,
               0.73,   0.75,   0.02, -13.82,  14.19,
              -0.26,   0.44,  -0.59,  -0.34,  -3.43,
            }).Reshape(n, n);
            AssertArray.AreAlmostEqual(luFactorization, NN.Array(a).Reshape(n, n), 1e-2, 1e-2);

            // Pivot indices
            AssertArray.AreEqual(new[] { 5, 5, 3, 4, 5 }, ipiv);
        }

        [TestMethod]
        public void TestSvd()
        {
            // LAPACKE_dgesvd (row-major, high-levell)
            // https://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/lapacke_dgesvd_row.c.htm
            var a = NN.Array(new[,] {
                {  8.79,  9.93,  9.83,  5.45,  3.16 },
                {  6.11,  6.91,  5.04, -0.27,  7.98 },
                { -9.15, -7.93,  4.86,  4.85,  3.01 },
                {  9.57,  1.64,  8.83,  0.74,  5.80 },
                { -3.49,  4.02,  9.80, 10.00,  4.27 },
                {  9.84,  0.15, -8.99, -6.02, -5.31 }
            });
            double[] s;
            Array<double> u, vt;
            Svd(a, out u, out s, out vt);

            var singularValues = NN.Array(new[] { 27.47, 22.64, 8.56, 5.99, 2.01 });
            AssertArray.AreAlmostEqual(singularValues, NN.Array(s), 1e-2, 1e-2);

            // Left singular vectors (stored columnwise)
            var leftSingularVectors = NN.Array(new[,] {
                { -0.59,   0.26,   0.36,   0.31,   0.23,   0.55 },
                { -0.40,   0.24,  -0.22,  -0.75,  -0.36,   0.18 },
                { -0.03,  -0.60,  -0.45,   0.23,  -0.31,   0.54 },
                { -0.43,   0.24,  -0.69,   0.33,   0.16,  -0.39 },
                { -0.47,  -0.35,   0.39,   0.16,  -0.52,  -0.46 },
                { 0.29,   0.58,  -0.02,   0.38,  -0.65,   0.11 }
            });
            AssertArray.AreAlmostEqual(leftSingularVectors, u, 1e-2, 1e-2);

            // Right singular vectors (stored rowwise)
            var rightSingularVectors = NN.Array(new[,] {
                { -0.25,  -0.40,  -0.69,  -0.37,  -0.41 },
                {  0.81,   0.36,  -0.25,  -0.37,  -0.10 },
                { -0.26,   0.70,  -0.22,   0.39,  -0.49 },
                {  0.40,  -0.45,   0.25,   0.43,  -0.62 },
                { -0.22,   0.14,   0.59,  -0.63,  -0.44 }
            });
            AssertArray.AreAlmostEqual(rightSingularVectors, vt, 1e-2, 1e-2);

            var sigma = NN.Zeros<double>(a.Shape[0], a.Shape[1]);
            sigma[0..s.Length, 0..s.Length] = NN.Diag(s);

            // Final check: A = U.Sigma.Vt
            AssertArray.AreAlmostEqual(a, u.Dot(sigma).Dot(vt));
        }

        private static void Svd(Array<double> a, out Array<double> u, out double[] s, out Array<double> vt)
        {
            var m = a.Shape[0];
            var n = a.Shape[1];
            var k = Math.Min(m, n);
            var superb = new double[k - 1];     // when matrix is ill-conditioned
            s = new double[k];
            u = NN.Zeros<double>(m, m);
            vt = NN.Zeros<double>(n, n);
            var copy = (double[])a.Values.Clone(); // if (jobu != 'O' && jobv != 'O') a is destroyed by dgesdv (https://software.intel.com/en-us/node/521150)
            Lapack.gesvd('A', 'A', m, n, copy, n, s, u.Values, m, vt.Values, n, superb);
        }

        //[TestMethod]
        //public void TestLsa()
        //{
        //    // LAPACKE_dgesvd (row-major, high-levell)
        //    // https://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/lapacke_dgesvd_row.c.htm
        //    var a = NN.Array(new double[,] {
        //        { 3, 4, 1, 0 },
        //        { 4, 3, 0, 1 },
        //        { 3, 4, 4, 3 },
        //        { 0, 1, 4, 3 },
        //        { 2, 0, 3, 3 },
        //        { 0, 1, 3, 4 }
        //    });
        //    var m = a.Shape[0];
        //    var n = a.Shape[1];

        //    double[] s = new double[n];
        //    double[] superb = new double[Math.Min(m, n) - 1];

        //    var u = NN.Zeros<double>(m, m);
        //    var vt = NN.Zeros<double>(n, n);
        //    double[] copy = (double[])a.Values.Clone(); // if (jobu != 'O' && jobv != 'O') a is destroyed by dgesdv (https://software.intel.com/en-us/node/521150)
        //    Lapack.dgesvd('A', 'A', m, n, copy, n, s, u.Values, m, vt.Values, n, superb);

        //    var singularValues = NN.Array(new[] { 27.47, 22.64, 8.56, 5.99, 2.01 });
        //    AssertArray.AreAlmostEqual(singularValues, NN.Array(s), 1e-2, 1e-2);

        //    // Left singular vectors (stored columnwise)
        //    var leftSingularVectors = NN.Array(new[,] {
        //        { -0.33, -0.53 },
        //        { -0.32, -0.54 },
        //        { -0.62, -0.10 },
        //        { -0.38,  0.42 },
        //        { -0.36,  0.25 },
        //        { -0.37,  0.42 }
        //    });
        //    AssertArray.AreAlmostEqual(leftSingularVectors, u, 1e-2, 1e-2);

        //    // Right singular vectors (stored rowwise)
        //    var rightSingularVectors = NN.Array(new[,] {
        //        { -0.42, -0.56 },
        //        { -0.48, -0.52 },
        //        { -0.57,  0.45 },
        //        { -0.51,  0.46 }
        //    });
        //    AssertArray.AreAlmostEqual(rightSingularVectors, vt, 1e-2, 1e-2);

        //    var sigma = NN.Zeros<double>(m, n);
        //    sigma[(0, n), (0, n)] = NN.Diag(s);

        //    // Final check: A = U.Sigma.Vt
        //    AssertArray.AreAlmostEqual(a, u.Dot(sigma).Dot(vt));
        //}

        [TestMethod]
        public void TestPseudoInverse()
        {
            // https://en.wikipedia.org/wiki/Moore–Penrose_pseudoinverse
            var a = NN.Array(new[,] {
               {  8.79,  9.93,  9.83,  5.45,  3.16 },
               {  6.11,  6.91,  5.04, -0.27,  7.98 },
               { -9.15, -7.93,  4.86,  4.85,  3.01 },
               {  9.57,  1.64,  8.83,  0.74,  5.80 },
               { -3.49,  4.02,  9.80, 10.00,  4.27 },
               {  9.84,  0.15, -8.99, -6.02, -5.31 }
            });

            var pseudoInv = PseudoInv(a);

            // https://en.wikipedia.org/wiki/Moore–Penrose_pseudoinverse
            AssertArray.AreAlmostEqual(a, a.Dot(pseudoInv).Dot(a));
            AssertArray.AreAlmostEqual(pseudoInv, pseudoInv.Dot(a).Dot(pseudoInv));

            // when a has linearly independent rows
            //AssertArray.AreAlmostEqual(NN.Eye<double>(a.Shape[0]), a.Dot(pseudoInv));

            // when a has linearly independent columns
            AssertArray.AreAlmostEqual(NN.Eye<double>(a.Shape[1]), pseudoInv.Dot(a));
        }

        [TestMethod]
        public void TestPseudoInverse2()
        {
            var a = NN.Array(new double[,] {
                { 1, 2, 3 },
                { 4, 5, 6 }
            });

            var pseudoInv = PseudoInv(a);

            // https://en.wikipedia.org/wiki/Moore–Penrose_pseudoinverse
            AssertArray.AreAlmostEqual(a, a.Dot(pseudoInv).Dot(a));
            AssertArray.AreAlmostEqual(pseudoInv, pseudoInv.Dot(a).Dot(pseudoInv));

            // when a has linearly independent rows
            AssertArray.AreAlmostEqual(NN.Eye<double>(a.Shape[0]), a.Dot(pseudoInv));

            // when a has linearly independent columns
            //AssertArray.AreAlmostEqual(NN.Eye<double>(a.Shape[1]), pseudoInv.Dot(a));
        }

        public static Array<double> PseudoInv(Array<double> a)
        {
            // https://en.wikipedia.org/wiki/Moore–Penrose_pseudoinverse
            // http://vene.ro/blog/inverses-pseudoinverses-numerical-issues-speed-symmetry.html
            var m = a.Shape[0];
            var n = a.Shape[1];

            /* Compute SVD */
            var k = Math.Min(m, n);
            double[] s = new double[k];
            var u = NN.Zeros<double>(m, m);
            var vt = NN.Zeros<double>(n, n);
            double[] copy = (double[])a.Values.Clone(); // if (jobu != 'O' && jobv != 'O') a is destroyed by dgesdv (https://software.intel.com/en-us/node/521150)
            double[] superb = new double[k - 1];
            Lapack.gesvd('A', 'A', m, n, copy, n, s, u.Values, m, vt.Values, n, superb);

            var invSigma = NN.Zeros<double>(n, m);
            invSigma[0..k, 0..k] = NN.Diag(1 / NN.Array(s));

            var pseudoInv = vt.T.Dot(invSigma).Dot(u.T);
            return pseudoInv;
        }

        // https://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/lapacke_dgels_row.c.htm
        [TestMethod]
        public void TestDgels()
        {
            /* Solve the equations A*X = B */
            const int M = 6;
            const int N = 4;
            const int NRHS = 2;
            const int LDA = N;
            const int LDB = NRHS;
            int m = M, n = N, nrhs = NRHS, lda = LDA, ldb = LDB;

            var a = new double[M * LDA] {
                1.44, -7.84, -4.39,  4.53,
               -9.96, -0.28, -3.24,  3.83,
               -7.55, 3.24, 6.27, -6.64,
                8.34, 8.09, 5.28,  2.06,
                7.08, 2.52, 0.74, -2.47,
               -5.45, -5.70, -1.19,  4.70
            };
            var b = new double[LDB * M] {
                8.58, 9.35,
                8.26, -4.43,
                8.48, -0.70,
               -5.28, -0.26,
                5.72, -7.36,
                8.93, -2.52
            };
            /* Solve the equations A*X = B */
            Lapack.gels('N', m, n, nrhs, a, lda, b, ldb);

            // Solution
            var solution = NN.Array<double>(new[,] {
                { -0.45,  0.25 },
                { -0.85, -0.90 },
                {  0.71,  0.63 },
                {  0.13,  0.14 }
            });
            var x = NN.Array(b).Reshape(-1, nrhs)[..n];
            AssertArray.AreAlmostEqual(solution, x, 1e-2, 1e-2);

            // Residual sum of squares for the solution
            var residual = new[] { 195.36, 107.06 };
            var bnm = NN.Array(b).Reshape(-1, nrhs)[n..m];
            for (int i = 0; i < residual.Length; i++)
            {
                AssertArray.AreAlmostEqual(residual[i], NN.NormSqr(bnm[i]));
            }

            // Details of QR factorization
            var details = NN.Array(new[,] {
                { -17.54, -4.76, -1.96,  0.42 },
                { -0.52,  12.40,  7.88, -5.84 },
                { -0.40,  -0.14, -5.75,  4.11 },
                {  0.44,  -0.66, -0.20, -7.78 },
                {  0.37,  -0.26, -0.17, -0.15 },
                { -0.29,   0.46,  0.41,  0.24 }
            });
            AssertArray.AreAlmostEqual(details, NN.Array(a).Reshape(M, LDA), 1e-2, 1e-2);
        }
        [TestMethod]
        public void testEigenValues()
        {
            const int N = 5;
            const int LDA = 5;
            int n = N, lda = LDA;
            /* Local arrays */
            double[] w = new double[N];
            double[] a = new double[]{
                1.96, -6.49, -0.47, -7.20, -0.65,
                0.00,  3.80, -6.39,  1.50, -6.34,
                0.00,  0.00, 4.17, -1.51, 2.67,
                0.00,  0.00, 0.00,  5.70, 1.80,
                0.00,  0.00, 0.00,  0.00, -7.10
            };
            Lapack.syev('V', 'U', n, a, lda, w);
            double[] trueEigenValues = new double[] { -11.07, -6.23, 0.86, 8.87, 16.09 };
            AssertArray.AreAlmostEqual(w, trueEigenValues, 1e-2, 1e-2);
        }

        //[TestMethod]
        //public void testEigenVectors()
        //{
        //    const int N = 5;
        //    int n = N;
        //    /* Local arrays */
        //    double[] w = new double[N];
        //    double[] a = new double[]{
        //        1.96, -6.49, -0.47, -7.20, -0.65,
        //        0.00,  3.80, -6.39,  1.50, -6.34,
        //        0.00,  0.00, 4.17, -1.51, 2.67,
        //        0.00,  0.00, 0.00,  5.70, 1.80,
        //        0.00,  0.00, 0.00,  0.00, -7.10
        //    };
        //    var b = Lapack.EigenVectors(a, n, 1, 'U');
        //    Assert.AreEqual(b[0], 0.8f);
        //}
    }
}