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
using System.Linq;
using System.IO;

using Proxem.BlasNet;
using Real = System.Single;

namespace Proxem.NumNet.Single
{
    using static ShapeUtil;

    public static class ArrayExtensions
    {
        public static Array<Real> Copy(this Array<Real> a, Array<Real> result = null)
        {
            if (a == result) return result;
            if (result == null) result = new Array<Real>(a.Shape);
            Array_.ElementwiseOp(result, a, (n, x, offsetx, incx, y, offsety, incy) =>
            {
                Blas.copy(n, y, offsety, incy, x, offsetx, incx);
            });
            return result;
        }

        #region Elementwise operations
		
		public static Array<Real> Equal(this Array<Real> a, Array<Real> b, Array<Real> result = null)
        {
            return Array_.ElementwiseOp(a, b, result,
                (n, x, offsetx, incx, y, offsety, incy, z, offsetz, incz) =>
                {
                    for (int i = 0; i < n; i++)
                    {
                        z[offsetz] = x[offsetx] == y[offsety] ? 1 : 0;
                        offsetx += incx;
                        offsety += incy;
                        offsetz += incz;
                    }
                });
        }

        public static Array<Real> Equal(this Array<Real> a, Real b, Array<Real> result = null)
        {
            if (result == null)
            {
                result = new Array<Real>(a.Shape);
            }
            Array_.ElementwiseOp(a, result,
                (n, x, offsetx, incx, z, offsetz, incz) =>
                {
                    for (int i = 0; i < n; i++)
                    {
                        z[offsetz] = x[offsetx] == b ? 1 : 0;
                        offsetx += incx;
                        offsetz += incz;
                    }
                });
            return result;
        }

        public static Array<Real> NotEqual(this Array<Real> a, Array<Real> b, Array<Real> result = null)
        {
            return Array_.ElementwiseOp(a, b, result,
                (n, x, offsetx, incx, y, offsety, incy, z, offsetz, incz) =>
                {
                    for (int i = 0; i < n; i++)
                    {
                        z[offsetz] = x[offsetx] != y[offsety] ? 1 : 0;
                        offsetx += incx;
                        offsety += incy;
                        offsetz += incz;
                    }
                });
        }

        public static Array<Real> NotEqual(this Array<Real> a, Real b, Array<Real> result = null)
        {
            if (result == null)
            {
                result = new Array<Real>(a.Shape);
            }
            Array_.ElementwiseOp(a, result,
                (n, x, offsetx, incx, z, offsetz, incz) =>
                {
                    for (int i = 0; i < n; i++)
                    {
                        z[offsetz] = x[offsetx] != b ? 1 : 0;
                        offsetx += incx;
                        offsetz += incz;
                    }
                });
            return result;
        }
		
        // result = a + alpha * b
        public static Array<Real> Add(this Array<Real> a, Array<Real> b, Real alpha = 1, Array<Real> result = null)
        {
            return Array_.ElementwiseOp(a, b, result,
                (n, x, offsetx, incx, y, offsety, incy, z, offsetz, incz) =>
            {
                if (alpha == 1 && incx == 1 && incy == 1 && incz == 1) Blas.vadd(n, x, offsetx, y, offsety, z, offsetz);
                else if (alpha == -1 && incx == 1 && incy == 1 && incz == 1) Blas.vsub(n, x, offsetx, y, offsety, z, offsetz);
                else if (z == x) Blas.axpy(n, alpha, y, offsety, incy, x, offsetx, incx);
                else if (z == y && alpha == 1) Blas.axpy(n, alpha, x, offsetx, incx, y, offsety, incy);
                // TODO: else if (incx == 0) => broadcast x ??
                // TODO: else if (incy == 0) => broadcast y ??
                else
                {
                    for (int i = 0; i < n; i++)         // TODO: Blas.copy y => x, Blas.axpy(1, x, y)
                    {
                        z[offsetz] = x[offsetx] + alpha * y[offsety];
                        offsetx += incx;
                        offsety += incy;
                        offsetz += incz;
                    }
                }   // See also: mkl_?omatadd => C := alpha*op(A) + beta*op(B)
            });
        }
		
		public static Array<Real> Add(this Array<Real> a, Real b, Real alpha = 1, Array<Real> result = null)
        {
            if (result == null)
            {
                result = new Array<Real>(a.Shape);
            }
            Array_.ElementwiseOp(a, result,
                (n, x, offsetx, incx, z, offsetz, incz) =>
                {
                    if (z == x) Blas.axpy(n, alpha, ref b, x, offsetx, incx);
                    else
                    {
                        for (int i = 0; i < n; i++)
                        {
                            z[offsetz] = x[offsetx] + alpha * b;
                            offsetx += incx;
                            offsetz += incz;
                        }
                    }
                });
            return result;
        }

        // result = a - alpha * b
        public static Array<Real> Sub(this Array<Real> a, Array<Real> b, Real alpha = 1, Array<Real> result = null) => a.Add(b, -alpha, result);
		
        public static Array<Real> Sub(this Array<Real> a, Real b, Real alpha = 1, Array<Real> result = null) => a.Add(b, -alpha, result);
		

        public static Array<Real> Div(this Array<Real> a, Array<Real> b, Array<Real> result = null)
        {
            return Array_.ElementwiseOp(a, b, result,
                (n, x, offsetx, incx, y, offsety, incy, z, offsetz, incz) =>
            {
                if (incx == 1 && incy == 1 && incz == 1) Blas.vdiv(n, x, offsetx, y, offsety, z, offsetz);
                // TODO: else if (incx == 0) dgemv 1/x
                // TODO: else if (incy == 0) dgemv 1/y
                else
                {
                    for (int i = 0; i < n; i++)
                    {
                        z[offsetz] = x[offsetx] / y[offsety];
                        offsetx += incx;
                        offsety += incy;
                        offsetz += incz;
                    }
                }
            });
        }

        /// <summary>
        /// result[i] = alpha / a[i]
        /// </summary>
        /// <param name="a"></param>
        /// <param name="alpha"></param>
        /// <param name="result"></param>
        /// <returns></returns>
        public static Array<Real> Inv(this Array<Real> a, Real alpha = 1, Array<Real> result = null)
        {
            return a.Apply(x => alpha / x, result: result);
        }

        public static Array<Real> Mul(this Array<Real> a, Array<Real> b, Array<Real> result = null)
        {
            return Array_.ElementwiseOp(a, b, result,
                (n, x, offsetx, incx, y, offsety, incy, z, offsetz, incz) =>
            {
                if (incx == 1 && incy == 1 && incz == 1) Blas.vmul(n, x, offsetx, y, offsety, z, offsetz);
                else if (incx == 0)   // x[offsetx] is broadcast: y[:] * x[offsetx]
                    Blas.gemv(Order.RowMajor, Transpose.NoTrans, n, 1, 1, y, offsety, incy, x, offsetx, 1, 0, z, offsetz, incz);
                //else if (incy == 0)   // y[offsety] is broadcast: x[:] * y[offsety]
                //    Blas.gemv(Order.RowMajor, Transpose.NoTrans, n, 1, 1, x, offsetx, incx, y, offsety, 1, 0, z, offsetz, incz);
                else    // when everything else fails, fallback to slow version
                {
                    for (int i = 0; i < n; i++)
                    {
                        z[offsetz] = x[offsetx] * y[offsety];
                        offsetx += incx;
                        offsety += incy;
                        offsetz += incz;
                    }
                }
            });
        }

        public static Array<Real> Clear(this Array<Real> a)
        {
            return a.Scale(0, result: a);      // TODO: temp
        }

        public static Array<Real> Scale(this Array<Real> a, Real alpha, Array<Real> result = null)
        {
            if (result != a) result = a.Copy(result: result);
            Array_.ElementwiseOp(0, result, 0, (n, x, offsetx, incx) => { Blas.scal(n, alpha, x, offsetx, incx); });
            return result;
        }

        public static Real Sum(this Array<Real> a)
        {
            Real result = 0;
            Array_.ElementwiseOp(0, a, 0,
                (n, x, offsetx, incx) =>
            {
                Real s = 0;
                for (int i = 0; i < n; i++)
                {
                    s += x[offsetx];
                    offsetx += incx;
                }
                result += s;
            });
            return result;
        }

        public static Array<Real> Sum(this Array<Real> a, int axis, Func<Real, Real> f, Array<Real> result = null, bool keepDims = false)
        {
            if (axis < 0) axis = a.Shape.Length + axis;

            result = result == null ? NN.Zeros<Real>(GetAggregatorResultShape(a, axis, true)) : result.Clear();

            var slice = a.Slices(); // TODO: create a new ElementwiseOp
            int ndim = a.Shape[axis];
            for (int d = 0; d < ndim; ++d)
            {
                slice[axis] = (d, d+1);
                Array_.ElementwiseOp(a[slice], result, (n, _a, off_a, step_a, _r, off_r, step_r) => {
                    Real s = 0;
                    for (int i = 0; i < n; i++)
                    {
                        s += f(_a[off_a]);
                        off_a += step_a;
                        off_r += step_r;
                    }
                    result += s;
                });
            }
            if (!keepDims)
                result = result.Reshape(RemoveAxis(result.Shape, axis));
            return result;
        }

        /// <summary> a.Sum(axis = 1)[i] = α * Σ_j a[i, j] </summary>
        public static Array<Real> Sum(this Array<Real> a, int axis, Array<Real> result = null, Real alpha = 1, Real beta = 0, bool keepDims = false)
        {
            if (axis < 0) axis = a.Shape.Length + axis;

            if (beta != 1 && result != null)
                result.Scale(beta, result: result);

            result = result ?? NN.Zeros<Real>(GetAggregatorResultShape(a, axis, keepDims));

            foreach (var row in a.UnsafeRows(axis, keepDims))
                result.Acc(row, alpha: alpha);

            return result;
        }

        public static Real Max(this Array<Real> a)
        {
            Real result = Real.NegativeInfinity;
            Array_.ElementwiseOp(0, a, 0,
                (n, x, offsetx, incx) =>
            {
                Real s = Real.NegativeInfinity;
                for (int i = 0; i < n; i++)
                {
                    s = Math.Max(s, x[offsetx]);
                    offsetx += incx;
                }
                result = Math.Max(result, s);
            });
            return result;
        }

        public static Array<Real> Max(this Array<Real> a, int axis, bool keepDims = false, Array<Real> result = null)
        {
            if (axis < 0) axis = a.Shape.Length + axis;

            result = result ?? NN.Zeros<Real>(GetAggregatorResultShape(a, axis, keepDims));

            bool firstTime = true;
            foreach (var row in a.UnsafeRows(axis, keepDims))
            {
                if (firstTime)
                {
                    result._ = row;
                    firstTime = false;
                }
                else
                    NN.Apply(result, row, (x, y) => Math.Max(x, y), result: result);
            }

            return result;
        }

        public static Real Min(this Array<Real> a)
        {
            Real result = Real.PositiveInfinity;
            Array_.ElementwiseOp(0, a, 0,
                (n, x, offsetx, incx) =>
            {
                Real s = Real.PositiveInfinity;
                for (int i = 0; i < n; i++)
                {
                    s = Math.Min(s, x[offsetx]);
                    offsetx += incx;
                }
                result = Math.Min(result, s);
            });
            return result;
        }

        public static Array<Real> Min(this Array<Real> a, int axis, bool keepDims = false, Array<Real> result = null)
        {
            if (axis < 0) axis = a.Shape.Length + axis;
            result = result ?? NN.Zeros<Real>(GetAggregatorResultShape(a, axis, keepDims));

            bool firstTime = true;
            foreach (var row in a.UnsafeRows(axis, keepDims))
            {
                if (firstTime)
                {
                    result._ = row;
                    firstTime = false;
                }
                else
                    NN.Apply(result, row, (x, y) => Math.Min(x, y), result: result);
            }

            return result;
        }

        public static Real Mean(this Array<Real> a)
        {
            return a.Sum() / a.Size;
        }

        public static int Argmax(this Array<Real> a)
        {
            Real min = Real.NegativeInfinity;
            int pos = -1;
            Array_.ElementwiseOp(0, a, 0,
                (n, x, offsetx, incx) =>
            {
                for (int i = 0; i < n; i++)
                {
                    if (x[offsetx] > min)
                    {
                        min = x[offsetx];
                        pos = offsetx;
                    }
                    offsetx += incx;
                }
            });
            return pos - a.Offset;
        }

        /// <summary><a href="http://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html">see NumPy doc</a></summary>
        public static Array<int> Argmax(this Array<Real> a, int axis, bool keepDims = false, Array<int> result = null)
        {
            if (axis < 0) axis = a.Shape.Length + axis;
            if (result == null)
                result = NN.Zeros<int>(GetAggregatorResultShape(a, axis, true));
            else if (result.NDim != a.NDim)
                result = result.Reshape(GetAggregatorResultShape(a, axis, true));

            Array_.ElementwiseOp(a, result, (n, x, offx, incx, y, offy, incy) =>
            {
                var max = x[offx]; offx += incx;
                var argmax = 0;
                for (int i = 1; i < n; ++i)
                {
                    var value = x[offx];
                    if (value > max)
                    {
                        max = value;
                        argmax = i;
                    }
                    offx += incx;
                }
                y[offy] = argmax;
            }, axis);

            return keepDims ? result : result.Reshape(GetAggregatorResultShape(a, axis, false));
        }

        public static int Argmin(this Array<Real> a)
        {
            Real min = Real.PositiveInfinity;
            int pos = -1;
            Array_.ElementwiseOp(0, a, 0,
                (n, x, offsetx, incx) =>
            {
                for (int i = 0; i < n; i++)
                {
                    if (x[offsetx] < min)
                    {
                        min = x[offsetx];
                        pos = offsetx;
                    }
                    offsetx += incx;
                }
            });
            return pos;
        }

        public static Array<int> Argmin(this Array<Real> a, int axis, bool keepDims = false, Array<int> result = null)
        {
            // http://docs.scipy.org/doc/numpy/reference/generated/numpy.argmin.html
            if (axis < 0) axis = a.Shape.Length + axis;
            result = result ?? NN.Zeros<int>(GetAggregatorResultShape(a, axis, keepDims));

            var slice = a.Slices(); // TODO: create a new ElementwiseOp
            slice[axis] = 0;
            var target = keepDims ? result[slice] : result;
            var max = a[slice].Copy();
            for (int d = 1; d < a.Shape[axis]; ++d)
            {
                slice[axis] = d;
                var row = a[slice];
                for (int i = 0; i < target.Shape[0]; i++)
                {
                    if (row.Item[i] < max.Item[i])
                    {
                        max.Item[i] = row.Item[i];
                        target.Item[i] = d;
                    }
                }
            }

            return result;
        }

        public static Array<Real> Normalize(this Array<Real> a, Array<Real> result = null)
        {
            return a.Scale(1 / NN.Norm(a), result: result);
        }

        /// <summary>a += α * b </summary>
        public static void Acc(this Array<Real> a, Array<Real> b, Real alpha = 1)
        {
            if (alpha == 0) return;
            b.AssertOfShape(a.Shape);
            Array_.ElementwiseOp(b, a, (n, x, offsetx, incx, y, offsety, incy) =>
            {
                // TODO: broadcasting ?
                Blas.axpy(n, alpha, x, offsetx, incx, y, offsety, incy);
            });
        }
        #endregion

        public static Real VectorDot(this Array<Real> a, Array<Real> b)
        {
            int lengtha = a.Shape.Length;
            int n = a.Shape[lengtha - 1];

            int inca;
            if (a.Shape.Length == 1)
            {
                inca = a.Stride[0];
            }
            else if (a.Shape.Length == 2)
            {
                if (a.Shape[0] != 1) throw new ArgumentException();
                inca = a.Stride[1];
            }
            else throw new NotImplementedException();

            int incb;
            if (b.Shape.Length == 1)
            {
                if (b.Shape[0] != n) throw new ArgumentException();
                incb = b.Stride[0];
            }
            else if (b.Shape.Length == 2)
            {
                if (b.Shape[0] != 1) throw new ArgumentException();
                if (b.Shape[1] != n) throw new ArgumentException();
                incb = b.Stride[1];
            }
            else throw new NotImplementedException();

            return Blas.dot(n, a.Values, a.Offset, inca, b.Values, b.Offset, incb);
        }

        /// <summary>
        /// Matrix multiplication.
        /// Returns: alpha * dot(a, b) + beta * result
        /// Ie with default value: dot(a, b)
        /// </summary>
        /// <remarks>
        /// For 2-D arrays it is equivalent to matrix multiplication,
        /// and for 1-D arrays to inner product of vectors (without complex conjugation).
        /// For N dimensions it is a sum product over the last axis of a and the second-to-last of b:
        /// dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])
        /// `TensorDot` provides more control for N-dim array multiplication.
        /// </remarks>
        public static Array<Real> Dot(this Array<Real> a, Array<Real> b, Array<Real> result = null, Real alpha = 1, Real beta = 0, bool transA = false, bool transB = false)
        {
            if (transA)
            {
                if (a.Shape.Length == 1)
                {
                    if (b.Shape.Length == 1)
                    {
                        if (transB)
                            return a.Dot(b, result, alpha, beta);
                        else
                            return a.VectorDot(b);
                    }
                    if (b.Shape.Length != 2) throw new NotImplementedException();
                    return b.Dot(a, result, alpha, beta, transA: !transB, transB: false);
                }
                else
                {
                    a = a.T;    // TODO: optimize => avoid creating new shape, stride, ...
                    //if (b.Shape.Length == 1 && !transB) b = b.Reshape(1, b.Shape[0]);
                }
                transA = false;
            }

            if (transB)
            {
                if (b.Shape.Length == 1)
                {
                    if (a.Shape.Length == 1)
                    {
                        if (transA) throw new NotImplementedException();
                        return a.Outer(b, result: result, alpha: alpha, beta: 0);
                    }
                    throw new NotImplementedException();
                    //if (a.Shape.Length != 2) throw new NotImplementedException();
                    //if (a.IsTransposed())
                    //{
                    //    a = a.T;
                    //    transA = !transA;
                    //}
                    //result = new Array<Real>(transA ? a.Shape[1] : a.Shape[0], b.Shape[0]);   // TODO: result != null
                    //Blas.gemm(Order.RowMajor, transA ? Transpose.Trans : Transpose.NoTrans, Transpose.NoTrans,
                    //    result.Shape[0], result.Shape[1], 1,
                    //    alpha,
                    //    a.Values, a.Offset, a.Stride[0],
                    //    b.Values, b.Offset, b.Stride[0],
                    //    beta,
                    //    result.Values, result.Offset, result.Stride[0]);
                    //return result;
                }
                else
                {
                    b = b.T;    // TODO: optimize => avoid creating new shape, stride, ...
                }
                transB = false;
            }

            // TODO: check alpha & beta
            if (a.Shape.Length == 0) return b.Scale(a.Values[a.Offset]);
            if (b.Shape.Length == 0) return a.Scale(b.Values[b.Offset]);
            if (a.Shape.Length == 1)    // vector x tensor
            {
                if (b.Shape.Length == 1)    // vector x vector
                {
                    if (a.Shape[0] != b.Shape[0]) throw AssertArray.BadRank("objects are not aligned: [{0}] dot [{1}]", a.Shape[0], b.Shape[0]);
                    if (result == null)
                    {
                        result = new Array<Real>();
                    }
                    else
                    {
                        if (result.Shape.Length != 0) throw AssertArray.BadRank("objects are not aligned");
                    }

                    result.Values[result.Offset] = beta * result.Values[result.Offset] + alpha * Blas.dot(a.Shape[0], a.Values, a.Offset, a.Stride[0], b.Values, b.Offset, b.Stride[0]);
                    return result;
                }
                else if (b.Shape.Length == 2)   // vector x matrix
                {
                    if (a.Shape[0] != b.Shape[0]) throw new RankException("objects are not aligned");
                    if (result == null)
                    {
                        result = new Array<Real>(b.Shape[1]);
                    }
                    else
                    {
                        if (result.Shape.Length != 1) throw new RankException("objects are not aligned");
                        if (result.Shape[0] != b.Shape[1]) throw new RankException("objects are not aligned");
                    }

                    // dgemv computes matrix x vector => result = M.T.dot(v.T).T
                    transB = !transB;
                    if (b.IsTransposed())
                    {
                        transB = !transB;
                        b = b.T;
                    }
                    // y:= alpha * A' * x + beta * y
                    Blas.gemv(Order.RowMajor, transB ? Transpose.Trans : Transpose.NoTrans, b.Shape[0], b.Shape[1], alpha,
                        b.Values, b.Offset, b.Stride[0],
                        a.Values, a.Offset, a.Stride[0],
                        beta,
                        result.Values, result.Offset, result.Stride[0]);
                    return result;
                }
                else if (b.Shape.Length == 3) // vector x tensor3
                {
                    // TODO: beta ?
                    if (a.Shape[0] != b.Shape[1]) throw new RankException("objects are not aligned");
                    if (result == null)
                    {
                        result = new Array<Real>(b.Shape[0], b.Shape[2]);
                    }
                    else
                    {
                        if (result.Shape[0] != b.Shape[0]) throw new RankException("objects are not aligned");
                        if (result.Shape[1] != b.Shape[2]) throw new RankException("objects are not aligned");
                    }

                    var offsetk = b.Offset;
                    var k_0 = result.Offset;
                    for (var k = 0; k < result.Shape[0]; k++)       // result.Shape[0] == b.Shape[0]
                    {
                        var offsetm = offsetk;
                        var k_m = k_0;
                        for (var m = 0; m < result.Shape[1]; m++)   // result.Shape[1] == b.Shape[2]
                        {
                            result.Values[k_m] = alpha * Blas.dot(a.Shape[0], a.Values, a.Offset, a.Stride[0], b.Values, offsetm, b.Stride[1]); // a.Shape[axis] == b.Shape[1];
                            offsetm += b.Stride[2];
                            k_m += result.Stride[1];
                        }
                        offsetk += b.Stride[0];
                        k_0 += result.Stride[0];
                    }

                    return result;
                }
                throw new NotImplementedException();
            }
            else if (b.Shape.Length == 1)    // tensor x vector
            {
                if (a.Shape.Length == 2)    // matrix x vector
                {
                    if (a.Shape[1] != b.Shape[0]) throw new RankException("objects are not aligned");
                    if (result == null)
                    {
                        result =new Array<Real>(a.Shape[0]);
                    }
                    else
                    {
                        if (result.Shape.Length != b.Shape.Length) throw new RankException("objects are not aligned");
                        if (result.Shape[0] != a.Shape[0]) throw new RankException("objects are not aligned");
                        // TODO: check strides
                    }
                    if ((a.Flags & Flags.Transposed) != 0)
                    {
                        transA = !transA;
                        a = a.T;
                    }
                    // y:= A*x + beta*y
                    if (a.Stride[1] == 1)
                    {
                        Blas.gemv(Order.RowMajor, transA ? Transpose.Trans : Transpose.NoTrans, a.Shape[0], a.Shape[1], alpha,
                            a.Values, a.Offset, a.Stride[0],
                            b.Values, b.Offset, b.Stride[0],
                            beta,
                            result.Values, result.Offset, result.Stride[0]);
                    }
                    else
                    {
                        // y *= beta
                        if (beta != 1)
                            result.Scale(beta, result: result);

                        int offB = b.Offset;
                        int offA = a.Offset;
                        for (int j = 0; j < b.Shape[0]; ++j)
                        {
                            Blas.axpy(a.Shape[0], alpha * b.Values[offB],
                                a.Values, offA, a.Stride[0],
                                result.Values, result.Offset, result.Stride[0]);
                            offB += b.Stride[0];
                            offA += a.Stride[1];
                        }
                    }

                    return result;
                }
                else if (a.Shape.Length == 3)    // tensor x vector = mat
                {
                    if (a.Shape[2] != b.Shape[0]) throw new RankException("objects are not aligned");
                    if (result == null)
                    {
                        result = new Array<Real>(a.Shape[0], a.Shape[1]);
                    }
                    else if (result.Shape[0] != a.Shape[0] || result.Shape[1] != a.Shape[1])
                    {
                        throw new RankException("objects are not aligned");
                    }

                    var offsetk = a.Offset;
                    var offsetRes = result.Offset;

                    for (var k = 0; k < result.Shape[0]; k++)
                    {
                        var offsetj = offsetk;
                        for (var j = 0; j < result.Shape[1]; j++)
                        {
                            result.Values[offsetRes] = alpha * Blas.dot(a.Shape[2], a.Values, offsetj, a.Stride[2], b.Values, b.Offset, b.Stride[0]);
                            offsetj += a.Stride[1];
                            offsetRes += result.Stride[1];
                        }
                        offsetk += a.Stride[0];
                    }
                    return result;
                }
                throw new NotImplementedException();
            }
            else if (a.Shape.Length == 2 && b.Shape.Length == 2)    // matrix x matrix
            {
                if (a.Shape[1] != b.Shape[0]) throw AssertArray.BadRank("objects are not aligned: [{0}, {1}] dot [{2}, {3}]", a.Shape[0], a.Shape[1], b.Shape[0], b.Shape[1]);
                if (result == null)
                {
                    result = new Array<Real>(a.Shape[0], b.Shape[1]);
                }
                else
                {
                    if (result.Shape[0] != a.Shape[0] || result.Shape[1] != b.Shape[1])
                        throw AssertArray.BadRank("result target have incorrect shape: [{0}, {1}] instead of [{2}, {3}].", result.Shape[0], result.Shape[1], a.Shape[0], b.Shape[1]);
                    // TODO: check strides
                }
                var m = a.Shape[0];
                var n = b.Shape[1];
                var k = a.Shape[1];
                if ((a.Flags & Flags.Transposed) != 0)
                {
                    transA = !transA;
                    a = a.T;
                }
                if ((b.Flags & Flags.Transposed) != 0)
                {
                    transB = !transB;
                    b = b.T;
                }
                // C:= alpha * op(A) * op(B) + beta * C
                Blas.gemm(Order.RowMajor, transA ? Transpose.Trans : Transpose.NoTrans, transB ? Transpose.Trans : Transpose.NoTrans,
                    m, n, k,
                    alpha,
                    a.Values, a.Offset, a.Stride[0],
                    b.Values, b.Offset, b.Stride[0],
                    beta,
                    result.Values, result.Offset, result.Stride[0]);
                return result;
            }
            // tensor x tensor
            throw new NotImplementedException();
        }

        /// <summary>
        /// Compute the outer product of two vectors: result[i, j] = alpha * a[i] * b[j] + beta * result[i, j]
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="result"></param>
        /// <param name="alpha"></param>
        /// <param name="beta"></param>
        /// <returns></returns>
        public static Array<Real> Outer(this Array<Real> a, Array<Real> b, Array<Real> result = null, Real alpha = 1, Real beta = 0)
        {
            if (a.Shape.Length != 1 && (a.Shape.Length != 2 || a.Shape[1] != 1))
            {
                a = a.Reshape(a.Size);
            }
            if (b.Shape.Length != 1 && (b.Shape.Length != 2 || b.Shape[1] != 1))
            {
                b = b.Reshape(b.Size);
            }
            if (result == null)
            {
                result = new Array<Real>(a.Shape[0], b.Shape[0]);
            }
            else
            {
                if (result.Shape.Length != 2) throw new RankException("objects are not aligned");
                if (result.Shape[0] != a.Shape[0]) throw new RankException("objects are not aligned");
                if (result.Shape[1] != b.Shape[0]) throw new RankException("objects are not aligned");
                // TODO: check strides ?
            }
            Blas.gemm(Order.RowMajor, Transpose.NoTrans, Transpose.Trans, result.Shape[0], result.Shape[1], 1,
                alpha,
                a.Values, a.Offset, a.Stride[0],
                b.Values, b.Offset, b.Stride[0],
                beta,
                result.Values, result.Offset, result.Stride[0]);
            return result;
        }

        /// <summary>
        /// Tensor dot on axis = 0
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="result"></param>
        /// <returns></returns>
        public static Array<Real> TensorDot(this Array<Real> a, Array<Real> b, Array<Real> result = null, Real beta = 1)
        {
            if (a.Shape.Length == 1) return TensorDotLeft(a, b, result, beta);
            else return TensorDotRight(a, b, result, beta);
        }

        public static Array<Real> TensorDotLeft(this Array<Real> vec, Array<Real> mat, Array<Real> result = null, Real beta = 1)
        {
            if (vec.Shape.Length != 1) throw new ArgumentException();
            if (mat.Shape.Length != 2) throw new ArgumentException();

            var shape = new[] { vec.Shape[0], mat.Shape[0], mat.Shape[1] };
            if (result == null) result = new Array<Real>(shape);
            else if (result.Shape.Length != shape.Length) throw new ArgumentException();    // TODO: check axes

            var offsetRes = result.Offset;
            for (int i = 0; i < vec.Shape[0]; ++i)
            {
                //result[i, _, _] = vec[i] * mat + beta * result[i, _, _]
                if (beta != 1) Blas.scal(mat.Size, beta, mat.Values, offsetRes, mat.Stride[1]);
                Blas.axpy(mat.Size, vec.Item[i], mat.Values, mat.Offset, mat.Stride[1], result.Values, offsetRes, result.Stride[2]);
                offsetRes += result.Stride[0];
            }
            return result;
        }

        public static Array<Real> TensorDotRight(this Array<Real> mat, Array<Real> vec, Array<Real> result = null, Real beta = 1)
        {
            if (vec.Shape.Length != 1) throw new ArgumentException();
            if (mat.Shape.Length != 2) throw new ArgumentException();

            var shape = new[] { mat.Shape[0], mat.Shape[1], vec.Shape[0] };
            if (result == null) result = new Array<Real>(shape);
            else if (result.Shape.Length != shape.Length) throw new ArgumentException(); // TODO: check axes

            var offsetRes = result.Offset;
            for (int i = 0; i < vec.Shape[0]; ++i)
            {
                // result[_, _, i] = vec[i] * mat + beta * result[_, _, i]
                if (beta != 1) Blas.scal(mat.Size, beta, mat.Values, offsetRes, mat.Stride[1]);
                Blas.axpy(mat.Size, vec.Item[i], mat.Values, mat.Offset, mat.Stride[1], result.Values, offsetRes, result.Stride[1]);
                offsetRes += result.Stride[2];
            }
            return result;
        }

        /// <summary>
        /// Uses the tensor to combine x and y.
        /// It's a shortcut for t.dot(x).dot(y) = y.dot(t).dot(x)
        /// result = alpha * (T . x) . y + beta * result = alpha * (y . T) . x + beta * result
        /// </summary>
        /// <param name="t">The tensor used to combine x and y: t.Shape = [k, j, i]</param>
        /// <param name="x">A vector: x.Shape = [i]</param>
        /// <param name="y">A vector: y.Shape = [j]</param>
        /// <param name="result">A vector: result.Shape = [k], if null, will be created.</param>
        /// <returns>Returns result.</returns>
        public static Array<Real> Combine(this Array<Real> t, Array<Real> x, Array<Real> y, Array<Real> result = null, Real alpha = 1, Real beta = 0)
        {
            if (t.Shape.Length != 3 && x.Shape.Length != 1 && y.Shape.Length != 1)
                throw new ArgumentException();

            //dispatch between different CombineXX according how t has been transposed
            if (t.Stride[0] > t.Stride[1] && t.Stride[1] > t.Stride[2])  //[0, 1, 2], no transposition
            {
                return t.Combine21(x, y, result: result, alpha: alpha, beta: beta);
            }
            else if (t.Stride[1] > t.Stride[0] && t.Stride[0] > t.Stride[2])  //[1, 0, 2]
            {
                return t.Transpose(1, 0, 2).Combine20(x, y, result: result, alpha: alpha, beta: beta);
            }
            else if (t.Stride[1] > t.Stride[2] && t.Stride[2] > t.Stride[0])  //[2, 0, 1]
            {
                return t.Transpose(1, 2, 0).Combine10(x, y, result: result, alpha: alpha, beta: beta);
            }
            //symetric cases
            else if (t.Stride[0] > t.Stride[2] && t.Stride[2] > t.Stride[1])  //[0, 2, 1]
            {
                return t.Transpose(0, 2, 1).Combine21(y, x, result: result, alpha: alpha, beta: beta);
            }
            else if (t.Stride[2] > t.Stride[0] && t.Stride[0] > t.Stride[1])  //[1, 2, 0]
            {
                return t.Transpose(2, 0, 1).Combine20(y, x, result: result, alpha: alpha, beta: beta);
            }
            else if (t.Stride[2] > t.Stride[1] && t.Stride[1] > t.Stride[0])  //[2, 1, 0]
            {
                return t.Transpose(2, 1, 0).Combine10(y, x, result: result, alpha: alpha, beta: beta);
            }
            else
            {
                throw new NotImplementedException();
            }
        }

        /// <summary>
        /// The basic Combine version
        /// result = alpha * (T . x) . y + beta * result
        /// </summary>
        public static Array<Real> Combine21(this Array<Real> t, Array<Real> x, Array<Real> y, Array<Real> result = null, Real alpha = 1, Real beta = 0)
        {
            if (t.Shape[2] != x.Shape[0] && t.Shape[1] != y.Shape[0])
                throw new ArgumentException();
            if (t.Stride[2] != 1)
                throw new NotImplementedException();

            int offsetT = t.Offset;

            if (result == null)
                result = new Array<Real>(t.Shape[0]);
            else
                result.Scale(beta, result: result);

            int offsetRes = result.Offset;
            int strideRes = result.Stride[0];

            int offsetX = x.Offset;
            int strideX = x.Stride[0];

            int strideJ = y.Stride[0];
            int J = y.Shape[0] * strideJ + y.Offset;

            for (int j = y.Offset; j < J; j += strideJ)
            {
                // result += alpha * y[j] * (t[:, j, :] . x)
                Blas.gemv(Order.RowMajor, Transpose.NoTrans,
                    t.Shape[0], t.Shape[2],
                    alpha * y.Values[j],
                    t.Values, offsetT, t.Stride[0],
                    x.Values, offsetX, strideX,
                    1,
                    result.Values, offsetRes, strideRes
                );
                offsetT += t.Stride[1];
            }
            return result;
        }

        /// <summary>
        /// A modified Combine version.
        /// result = alpha * z . (T . x) + beta * result
        /// </summary>
        /// <param name="t">The tensor used to combine x and z: t.Shape = [k, j, i]</param>
        /// <param name="x">A vector: x.Shape = [i]</param>
        /// <param name="z">A vector: z.Shape = [k]</param>
        /// <param name="result">A vector: result.Shape = [j], if null, will be created.</param>
        /// <returns>Returns result.</returns>
        public static Array<Real> Combine20(this Array<Real> t, Array<Real> x, Array<Real> z, Array<Real> result = null, Real alpha = 1, Real beta = 0)
        {
            if (t.Shape.Length != 3 && x.Shape.Length != 1 && z.Shape.Length != 1)
                throw new ArgumentException();
            if (t.Shape[2] != x.Shape[0] && t.Shape[0] != z.Shape[0])
                throw new ArgumentException();
            if (t.Stride[2] != 1)
                throw new NotImplementedException();

            if (result == null)
                result = new Array<Real>(t.Shape[1]);
            else
                result.Scale(beta, result: result);

            int strideK = z.Stride[0];
            int K = z.Shape[0] * strideK + z.Offset;

            int strideT = t.Stride[0];
            int offsetT = t.Offset;

            for (int k = z.Offset; k < K; k += strideK)
            {
                // result += alpha * z[k] * (t[k, :, :] . x)
                Blas.gemv(Order.RowMajor, Transpose.NoTrans, t.Shape[1], t.Shape[2],
                        alpha * z.Values[k],
                        t.Values, offsetT, t.Stride[1],
                        x.Values, x.Offset, x.Stride[0],
                        1,
                        result.Values, result.Offset, result.Stride[0]);
                offsetT += t.Stride[0];
            }

            return result;
        }

        /// <summary>
        /// A modified Combine version.
        /// result = alpha * z . (y . T) + beta * result
        /// </summary>
        /// <param name="t">The tensor used to combine y and z: t.Shape = [k, j, i]</param>
        /// <param name="y">A vector: y.Shape = [j]</param>
        /// <param name="z">A vector: z.Shape = [k]</param>
        /// <param name="result">A vector: result.Shape = [i], if null, will be created.</param>
        /// <returns>Returns result.</returns>
        public static Array<Real> Combine10(this Array<Real> t, Array<Real> y, Array<Real> z, Array<Real> result = null, Real alpha = 1, Real beta = 0)
        {
            if (t.Shape.Length != 3 && y.Shape.Length != 1 && z.Shape.Length != 1)
                throw new ArgumentException();
            if (t.Shape[1] != y.Shape[0] && t.Shape[0] != z.Shape[0])
                throw new ArgumentException();
            if (t.Stride[2] != 1)
                throw new NotImplementedException();

            if (result == null)
                result = new Array<Real>(t.Shape[2]);
            else
                result.Scale(beta, result: result);

            int strideK = z.Stride[0];
            int K = z.Shape[0] * strideK + z.Offset;

            int strideT = t.Stride[0];
            int offsetT = t.Offset;

            for (int k = z.Offset; k < K; k += strideK)
            {
                // result += alpha * z[k] * (y . t[k, :, :])
                Blas.gemv(Order.RowMajor, Transpose.Trans, t.Shape[1], t.Shape[2],
                        alpha * z.Values[k],
                        t.Values, offsetT, t.Stride[1],
                        y.Values, y.Offset, y.Stride[0],
                        1,
                        result.Values, result.Offset, result.Stride[0]);
                offsetT += t.Stride[0];
            }

            return result;
        }

        public static void Save(this Array<Real> t, BinaryWriter writer)
        {
            writer.Write(t.Shape.Length);
            for (int axis = 0; axis < t.Shape.Length; axis++)
            {
                writer.Write(t.Shape[axis]);
                writer.Write(t.Stride[axis]);
            }
            writer.Write(t.Offset);
            writer.Write(t.Values.Length);
            for (int i = 0; i < t.Values.Length; i++)
            {
                writer.Write(t.Values[i]);
            }
        }

        public static Array<Real> Load(BinaryReader reader)
        {
            var n = reader.ReadInt32();
            var shape = new int[n];
            var stride = new int[n];
            for (int axis = 0; axis < n; axis++)
            {
                shape[axis] = reader.ReadInt32();
                stride[axis] = reader.ReadInt32();
            }
            var offset = reader.ReadInt32();
            var length = reader.ReadInt32();
            var values = new Real[length];
            for (int i = 0; i < length; i++)
            {
                values[i] = reader.ReadSingle();
            }
            return new Array<Real>(shape, values, offset, stride);
        }
    }
}
