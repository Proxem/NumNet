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

using Proxem.BlasNet;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

using Int = System.Int32;

namespace Proxem.NumNet.Int32
{
    using static ShapeUtil;

    public static class ArrayExtensions
    {
        public static Array<Int> Copy(this Array<Int> a, Array<Int> result = null)
        {
            if (result == null) result = new Array<Int>(a.Shape);
            Array_.ElementwiseOp(result, a, (n, x, offsetx, incx, y, offsety, incy) =>
            {
                //Blas.copy(n, y, offsety, incy, x, offsetx, incx);
                for(int i = 0; i < n; i++)
                {
                    x[offsetx] = y[offsety];
                    offsetx += incx;
                    offsety += incy;
                }
            });
            return result;
        }

        #region Elementwise operations
        // result = a + alpha * b
        public static Array<Int> Add(this Array<Int> a, Array<Int> b, Int alpha = 1, Array<Int> result = null)
        {
            if (result == a)
                return a.Acc(b, alpha);

            return Array_.ElementwiseOp(a, b, result,
                (n, x, offsetx, incx, y, offsety, incy, z, offsetz, incz) =>
            {
                for (int i = 0; i < n; i++)
                {
                    z[offsetz] = x[offsetx] + alpha * y[offsety];
                    offsetx += incx;
                    offsety += incy;
                    offsetz += incz;
                }
            });
        }

        // result = a - alpha * b
        public static Array<Int> Sub(this Array<Int> a, Array<Int> b, Int alpha = 1, Array<Int> result = null) => a.Add(b, -alpha, result);

        public static Array<Int> Div(this Array<Int> a, Array<Int> b, Array<Int> result = null)
        {
            return Array_.ElementwiseOp(a, b, result,
                (n, x, offsetx, incx, y, offsety, incy, z, offsetz, incz) =>
            {
                for (int i = 0; i < n; i++)
                {
                    z[offsetz] = x[offsetx] / y[offsety];
                    offsetx += incx;
                    offsety += incy;
                    offsetz += incz;
                }
            });
        }

        public static Array<Int> Inv(this Array<Int> a, Array<Int> result = null)
        {
            return a.Apply(x => 1 / x, result: result);
        }

        public static Array<Int> Mul(this Array<Int> a, Array<Int> b, Array<Int> result = null)
        {
            return Array_.ElementwiseOp(a, b, result,
                (n, x, offsetx, incx, y, offsety, incy, z, offsetz, incz) =>
            {
                for (int i = 0; i < n; i++)
                {
                    z[offsetz] = x[offsetx] * y[offsety];
                    offsetx += incx;
                    offsety += incy;
                    offsetz += incz;
                }
            });
        }

        public static Array<Int> Neq(this Array<Int> a, Array<Int> b, Int alpha = 1, Array<Int> result = null)
        {
            return Array_.ElementwiseOp(a, b, result,
                (n, x, offsetx, incx, y, offsety, incy, z, offsetz, incz) =>
            {
                for (int i = 0; i < n; i++)
                {
                    z[offsetz] = x[offsetx] ==  y[offsety] ? 0 : 1;
                    offsetx += incx;
                    offsety += incy;
                    offsetz += incz;
                }
            });
        }


        public static Array<Int> Clear(this Array<Int> a)
        {
            return a.Scale(0, result: a);      // TODO: temp
        }

        public static Array<Int> Scale(this Array<Int> a, Int alpha, Array<Int> result = null)
        {
            if (result != a) result = a.Copy(result: result);
            //ElementwiseOp(0, result, 0, (n, x, offsetx, incx) => { Blas.scal(n, alpha, x, offsetx, incx); });
            throw new NotImplementedException();
            //return result;
        }

        public static Array<Int> Apply(this Array<Int> a, Func<Int, Int> fn, Array<Int> result = null)
        {
            if (result == null) result = new Array<Int>(a.Shape);
            Array_.ElementwiseOp(a, result, (n, x, offsetx, incx, y, offsety, incy) =>
            {
                for (int i = 0; i < n; i++)
                {
                    y[offsety] = fn(x[offsetx]);
                    offsetx += incx;
                    offsety += incy;
                }
            });
            return result;
        }

        public static Array<Int> Apply(this Array<Int> a, Array<Int> b, Func<Int, Int, Int> fn, Array<Int> result = null)
        {
            return Array_.ElementwiseOp(a, b, result,
                (n, x, offsetx, incx, y, offsety, incy, z, offsetz, incz) =>
            {
                for (int i = 0; i < n; i++)
                {
                    z[offsetz] = fn(x[offsetx], y[offsety]);
                    offsetx += incx;
                    offsety += incy;
                    offsetz += incz;
                }
            });
        }

        public static Array<Int> Exp(this Array<Int> a, Array<Int> result = null)
        {
            return a.Apply(x => (Int)Math.Exp(x), result);
        }

        public static Array<Int> Pow(this Array<Int> a, double b, Array<Int> result = null)
        {
            return a.Apply(x => (Int)Math.Pow(x, b), result);
        }

        public static Array<Int> Tanh(this Array<Int> a, Array<Int> result = null)
        {
            return a.Apply(x => (Int)Math.Tanh(x), result);
        }

        public static Array<Int> Rectify(this Array<Int> a, Array<Int> result = null)
        {
            return a.Apply(x => x > 0 ? x : 0, result);
        }

        public static Array<Int> Sqrt(this Array<Int> a, Array<Int> result = null)
        {
            return a.Apply(x => (Int)Math.Sqrt(x), result);
        }

        public static Array<Int> Abs(this Array<Int> a, Array<Int> result = null)
        {
            return a.Apply(Math.Abs, result);
        }

        public static Int Sum(this Array<Int> a)
        {
            Int result = 0;
            Array_.ElementwiseOp(0, a, 0,
                (n, x, offsetx, incx) =>
            {
                Int s = 0;
                for (int i = 0; i < n; i++)
                {
                    s += x[offsetx];
                    offsetx += incx;
                }
                result += s;
            });
            return result;
        }

        public static Array<Int> Sum(this Array<Int> a, int axis, Array<Int> result = null, Int alpha = 1, Int beta = 0, bool keepDims = false)
        {
            if (axis < 0) axis = a.Shape.Length + axis;

            if (beta != 1 && result != null)
                result.Scale(beta, result: result);

            if (result == null)
            {
                int[] shape = new int[a.Shape.Length];
                Array.Copy(a.Shape, 0, shape, 0, a.NDim);
                shape[axis] = 1;
                result = NN.Zeros<Int>(shape);
            }

            var slice = a.Slices(); // TODO: create a new ElementwiseOp
            for (int d = 0; d < a.Shape[axis]; ++d)
            {
                slice[axis] = (d, d + 1);
                result.Acc(a[slice], alpha: alpha);
            }
            if (!keepDims)
            {
                var resultShape = new int[a.NDim - 1];
                Array.Copy(a.Shape, 0, resultShape, 0, axis);
                if (axis < a.NDim - 1)
                    Array.Copy(a.Shape, axis + 1, resultShape, axis, a.NDim - axis - 1);
                result = result.Reshape(resultShape);
            }
            return result;
        }

        public static Int Max(this Array<Int> a)
        {
            Int result = Int.MinValue;
            Array_.ElementwiseOp(0, a, 0,
                (n, x, offsetx, incx) =>
            {
                Int s = Int.MinValue;
                for (int i = 0; i < n; i++)
                {
                    s = Math.Max(s, x[offsetx]);
                    offsetx += incx;
                }
                result = Math.Max(result, s);
            });
            return result;
        }

        public static Array<Int> Max(this Array<Int> a, int axis, bool keepDims = false, Array<Int> result = null)
        {
            if (axis < 0) axis = a.Shape.Length + axis;
            if (result == null)
            {
                var extra = keepDims ? 1 : 0;
                int[] shape = new int[a.Shape.Length - 1 + extra];
                Array.Copy(a.Shape, 0, shape, 0, axis);
                if (keepDims) shape[axis] = 1;
                Array.Copy(a.Shape, axis + 1, shape, axis + extra, shape.Length - axis - extra);
                result = NN.Zeros<Int>(shape);
            }

            var slice = a.Slices(); // TODO: create a new ElementwiseOp
            slice[axis] = 0;
            var target = keepDims ? result[slice] : result;
            Apply(target, a[slice], (x, y) => y, result: target);

            for (int d = 1; d < a.Shape[axis]; ++d)
            {
                slice[axis] = d;
                Apply(target, a[slice], (x, y) => Math.Max(x, y), result: target);
            }
            return result;
        }


        public static Int Min(this Array<Int> a)
        {
            Int result = Int.MaxValue;
            Array_.ElementwiseOp(0, a, 0,
                (n, x, offsetx, incx) =>
            {
                Int s = Int.MaxValue;
                for (int i = 0; i < n; i++)
                {
                    s = Math.Min(s, x[offsetx]);
                    offsetx += incx;
                }
                result = Math.Min(result, s);
            });
            return result;
        }

        public static Array<Int> Min(this Array<Int> a, int axis, bool keepDims = false, Array<Int> result = null)
        {
            if (axis < 0) axis = a.Shape.Length + axis;
            if (result == null)
            {
                var extra = keepDims ? 1 : 0;
                int[] shape = new int[a.Shape.Length - 1 + extra];
                Array.Copy(a.Shape, 0, shape, 0, axis);
                if (keepDims) shape[axis] = 1;
                Array.Copy(a.Shape, axis + 1, shape, axis + extra, shape.Length - axis - extra);
                result = NN.Zeros<Int>(shape);
            }

            var slice = a.Slices(); // TODO: create a new ElementwiseOp
            slice[axis] = 0;
            var target = keepDims ? result[slice] : result;
            Apply(target, a[slice], (x, y) => y, result: target);

            for (int d = 1; d < a.Shape[axis]; ++d)
            {
                slice[axis] = d;
                Apply(target, a[slice], (x, y) => Math.Min(x, y), result: target);  // target is still result[slice[axis] == 0]
            }
            return result;
        }

        public static Int Mean(this Array<Int> a)
        {
            return a.Sum() / a.Size;
        }

        public static int Argmax(this Array<Int> a, int[] result = null)
        {
            Int min = Int.MinValue;
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
            return pos;
        }

        /// <summary><a href="http://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html">see NumPy doc</a></summary>
        public static Array<int> Argmax(this Array<int> a, int axis, bool keepDims = false, Array<int> result = null)
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

        public static int[] ArgMin(this Array<Int> a, int[] result = null)
        {
            Int min = Int.MaxValue;
            int pos = -1;
            Array_.ElementwiseOp(0, a, 0,
                (n, x, offsetx, incx) =>
            {
                for (int i = 0; i < n;
                i++)
                {
                    if (x[offsetx] < min)
                    {
                        min = x[offsetx];
                        pos = offsetx;
                    }
                    offsetx += incx;
                }
            });
            return a.UnravelIndex(pos, result);
        }

        public static Array<int> Argmin(this Array<Int> a, int axis, bool keepDims = false, Array<int> result = null)
        {
            // http://docs.scipy.org/doc/numpy/reference/generated/numpy.argmin.html
            if (axis < 0) axis = a.Shape.Length + axis;
            if (result == null)
            {
                var extra = keepDims ? 1 : 0;
                int[] shape = new int[a.Shape.Length - 1 + extra];
                Array.Copy(a.Shape, 0, shape, 0, axis);
                if (keepDims) shape[axis] = 1;
                Array.Copy(a.Shape, axis + 1, shape, axis + extra, shape.Length - axis - extra);
                result = NN.Zeros<int>(shape);
            }

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

        public static Array<Int> Acc(this Array<Int> a, Array<Int> b, Int alpha = 1)
        {
            Array_.ElementwiseOp(a, b,
            (n, x, offsetx, incx, y, offsety, incy) =>
            {
                for (int i = 0; i < n; i++)
                {
                    x[offsetx] += alpha * y[offsety];
                    offsetx += incx;
                    offsety += incy;
                }
            });
            return a;
        }
        #endregion

        public static void Save(this Array<Int> t, BinaryWriter writer)
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

        public static Array<Int> Load(BinaryReader reader)
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
            var values = new Int[length];
            for (int i = 0; i < length; i++)
            {
                values[i] = reader.ReadInt32();
            }
            return new Array<Int>(shape, values, offset, stride);
        }

        public static Slice[] Slices(this Array<Int> a)
        {
            Slice[] slices = new Slice[a.Shape.Length];
            for (int i = 0; i < slices.Length; ++i)
                slices[i] = (0, a.Shape[i]);
            return slices;
        }
    }

}
