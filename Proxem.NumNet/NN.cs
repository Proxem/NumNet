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
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Proxem.BlasNet;

namespace Proxem.NumNet
{
    /// <summary>
    /// Collection of methods to create and manipulate Arrays.
    /// </summary>
    public static partial class NN
    {
        public static Random Random = new Random();

        public static Array<T> Array<T>(params T[] values)
        {
            return (Array<T>)values;
        }

        public static Array<T> Array<T>(T[,] values)
        {
            return (Array<T>)values;
        }

        public static Array<T> Array<T>(T[,,] values)
        {
            return (Array<T>)values;
        }

        public static Array<T> Array<T>(T[,,,] values)
        {
            return (Array<T>)values;
        }

        public static Array<T> Array<T>(params T[][] values)
        {
            int n = values.Length;
            int m = values[0].Length;
            var flatValues = new T[n * m];
            for (int i = 0; i < n; i++)
            {
                System.Array.Copy(values[i], 0, flatValues, i * m, m);
            }
            return new Array<T>(new int[] { n, m }, flatValues);
        }

        public static Array<T> Array<T>(Array<T> a)
        {
            return a;
        }

        public static Array<float> Zeros(params int[] shape)
        {
            return Zeros<float>(shape);
        }

        public static Array<T> Zeros<T>(params int[] shape)
        {
            return new Array<T>(shape);
        }

        public static Array<float> Ones(params int[] shape)
        {
            return Ones<float>(shape);
        }

        public static Array<T> Ones<T>(params int[] shape)
        {
            var one = NumNet.Array<T>.Operators.Convert(1);
            var result = new Array<T>(shape);
            var values = result.Values;
            for (var i = 0; i < values.Length; i++)
                values[i] = one;
            return result;
        }

        public static Array<T> Const<T>(T a, params int[] shape)
        {
            var result = new Array<T>(shape);
            for (var i = 0; i < result.Values.Length; i++)
            {
                result.Values[i] = a;
            }
            return result;
        }

        public static Array<T> Fill<T>(Func<T> f, params int[] shape) => Fill(f, new Array<T>(shape));
        public static Array<T> Fill<T>(Func<T> f, Array<T> result)
        {
            return Array_.ElementwiseOp(result, (n, r, offR, strideR) =>
            {
                for (int i = 0; i < n; ++i)
                {
                    r[offR] = f();
                    offR += strideR;
                }
            });
        }

        public static Array<T> Scalar<T>(T a)
        {
            return new Array<T>(new int[]{}, new []{a});
        }

        public static Array<float> Eye(int n)
        {
            return Eye<float>(n);
        }

        public static Array<T> Eye<T>(int n)
        {
            var one = NumNet.Array<T>.Operators.Convert(1);
            var result = new Array<T>(n, n);
            for (var i = 0; i < n; i++)
            {
                result.Item[i, i] = one;
            }
            return result;
        }

        public static Array<float> Exchange(int n)
        {
            return Exchange<float>(n);
        }

        public static Array<T> Exchange<T>(int n)
        {
            var one = NumNet.Array<T>.Operators.Convert(1);
            var result = new Array<T>(n, n);
            for (var i = 0; i < n; i++)
            {
                result.Item[i, n - 1 - i] = one;
            }
            return result;
        }

        public static Array<T> Diag<T>(Array<T> v)
        {
            if (v.Shape.Length == 1)
            {
                var n = v.Shape[0];
                var result = new Array<T>(n, n);
                for (int i = 0; i < n; i++)
                {
                    result.Item[i, i] = v.Item[i];
                }
                return result;
            }
            if (v.Shape.Length == 2)
            {
                var n = v.Shape[0];
                if (v.Shape[1] != n) throw new ArgumentException("Square matrix expected");
                var result = new Array<T>(n);
                for (int i = 0; i < n; i++)
                {
                    result.Item[i] = v.Item[i, i];
                }
                return result;
            }
            throw new ArgumentException();
        }

        public static Array<T> Diag<T>(params T[] diag)
        {
            var n = diag.Length;
            var result = new Array<T>(n, n);
            for (int i = 0; i < n; i++)
            {
                result.Item[i, i] = diag[i];
            }
            return result;
        }

        public static Array<int> Range(int start, int stop, int step = 1)
        {
            return Range<int>(start, stop, step);
        }

        public static Array<T> Range<T>(int start, int stop, int step = 1, Array<T> result = null)
        {
            var length = (stop - start) / step;
            result?.AssertOfShape(length);
            result = result ?? new Array<T>(length);
            for (var i = 0; i < result.Shape[0]; i++)
            {
                result.Item[i] = NumNet.Array<T>.Operators.Convert(start);
                start += step;
            }
            return result;
        }

        public static Array<int> Range(int upper)
        {
            return Range<int>(upper);
        }

        public static Array<T> Range<T>(int upper)
        {
            return Range<T>(0, upper);
        }

        public static Array<T> OneHot<T>(int hot, int size)
        {
            var x = Zeros<T>(size);
            var one = NumNet.Array<T>.Operators.Convert(1);
            x.Item[hot] = one;
            return x;
        }

        public static Array<T> OneHot<T>(int[] hot, int[] shape)
        {
            var x = Zeros<T>(shape);
            var one = NumNet.Array<T>.Operators.Convert(1);
            x.Item[hot] = one;
            return x;
        }

        public static Array<float> OneHot(int hot, int size)
        {
            return OneHot<float>(hot, size);
        }

        public static Array<float> OneHot(int[] hot, int[] shape)
        {
            return OneHot<float>(hot, shape);
        }

        public static Array<T> Empty<T>()
        {
            return new Array<T>(EmptyArray<int>.Value, EmptyArray<T>.Value, 0, EmptyArray<int>.Value);
        }


        public static Array<T> Reshape<T>(Array<T> a, params int[] shape)
        {
            return a.Reshape(shape);
        }

        public static Array<T> MinifyDim<T>(Array<T> a)
        {
            return a.MinifyDim();
        }

        public static Array<T> Transpose<T>(Array<T> a, params int[] axes)
        {
            return a.Transpose(axes);
        }

        public static Array<T> Copy<T>(Array<T> a, Array<T> result = null) => a == result ? a : a.Apply(x => x, result);

        public static Array<T> Concat<T>(int axis, IEnumerable<Array<T>> inputs) => Concat(axis, inputs.ToArray());

        public static Array<T> Concat<T>(int axis, params Array<T>[] inputs)
        {
            if (inputs.Length < 1)
                throw new ArgumentException("Concat expects at least one array as argument");
            if (inputs.Length == 1)
                return inputs[0];
            if (axis < 0) axis += inputs[0].NDim;

            int ndim = inputs[0].NDim;
            var shape = new int[ndim];
            for (int a = 0; a < ndim; ++a)
                shape[a] = inputs[0].Shape[a];

            for (int i = 1; i < inputs.Length; ++i)
            {
#if DEBUG
                inputs[i].AssertOfDim(ndim);
                for (int a = 0; a < ndim; ++a)
                    if (a != axis && inputs[i].Shape[a] != shape[a])
                        throw new RankException($"Message can't concatenate [{string.Join(", ", shape)}] and [{string.Join(", ", inputs[i].Shape)}]");
#endif
                shape[axis] += inputs[i].Shape[axis];
            }

            var res = Zeros<T>(shape);
            return Concat(axis, inputs, res);
        }

        public static Array<T> Concat<T>(int axis, Array<T>[] inputs, Array<T> result)
        {
            var slices = result.Slices();
            slices[axis] = Slicer.Until(inputs[0].Shape[axis]);
            var view = result[slices];
            Copy(inputs[0], view);
            var stride = view.Stride[axis];

            for (int i = 1; i < inputs.Length; ++i)
            {
                view.Offset += stride * inputs[i - 1].Shape[axis];
                view.Shape[axis] = inputs[i].Shape[axis];
                Copy(inputs[i], view);
            }
            return result;
        }

        public static int MIN_SIZE_FOR_PARELLELISM = 65000;

        public static Array<R> Apply<T1, R>(this Array<T1> a, Func<T1, R> fn, Array<R> result = null)
        {
            if (result == null) result = new Array<R>(a.Shape);
            Array_.ElementwiseOp(a, result, (n, x0, offset0, inc0, x1, offset1, inc1) =>
            {
                if (n < MIN_SIZE_FOR_PARELLELISM)
                    for (int i = 0; i < n; i++)
                    {
                        x1[offset1] = fn(x0[offset0]);
                        offset0 += inc0;
                        offset1 += inc1;
                    }
                else
                {
                    var threads = Blas.NThreads;
                    Parallel.For(0, threads, t =>
                    {
                        var end = (n - 1) / threads + 1;
                        var off0 = offset0 + t * end * inc0;
                        var off1 = offset1 + t * end * inc0;
                        if (t == threads - 1) end = n - t * end;

                        for (int i = 0; i < end; i++)
                        {
                            x1[off1] = fn(x0[off0]);
                            off0 += inc0;
                            off1 += inc1;
                        }
                    });
                }
            });
            return result;
        }

        public static Array<R> Apply<T1, T2, R>(Array<T1> a, Array<T2> b, Func<T1, T2, R> fn, Array<R> result = null)
        {
            return Array_.ElementwiseOp(a, b, result,
                (n, x0, offset0, inc0, x1, offset1, inc1, x2, offset2, inc2) =>
                {
                    if (n < MIN_SIZE_FOR_PARELLELISM)
                        for (int i = 0; i < n; i++)
                        {
                            x2[offset2] = fn(x0[offset0], x1[offset1]);
                            offset0 += inc0;
                            offset1 += inc1;
                            offset2 += inc2;
                        }
                    else
                    {
                        var threads = Blas.NThreads;
                        Parallel.For(0, threads, t =>
                        {
                            var end = (n - 1) / threads + 1;
                            var off0 = offset0 + t * end * inc0;
                            var off1 = offset1 + t * end * inc1;
                            var off2 = offset2 + t * end * inc2;
                            if (t == threads - 1) end = n - t * end;

                            for (int i = 0; i < end; i++)
                            {
                                x2[off2] = fn(x0[off0], x1[off1]);
                                off0 += inc0;
                                off1 += inc1;
                                off2 += inc2;
                            }
                        });
                    }
                });
        }

        public static Array<R> Apply<T1, T2, T3, R>(Array<T1> a, Array<T2> b, Array<T3> c, Func<T1, T2, T3, R> fn, Array<R> result = null)
        {
            return Array_.ElementwiseOp(a, b, c, result,
                (n, x0, offset0, inc0, x1, offset1, inc1, x2, offset2, inc2, x3, offset3, inc3) =>
                {
                    if (n < MIN_SIZE_FOR_PARELLELISM)
                        for (int i = 0; i < n; i++)
                        {
                            x3[offset3] = fn(x0[offset0], x1[offset1], x2[offset2]);
                            offset0 += inc0;
                            offset1 += inc1;
                            offset2 += inc2;
                            offset3 += inc3;
                        }
                    else
                    {
                        var threads = Blas.NThreads;
                        Parallel.For(0, threads, t =>
                        {
                            var end = (n - 1) / threads + 1;
                            var off0 = offset0 + t * end * inc0;
                            var off1 = offset1 + t * end * inc1;
                            var off2 = offset2 + t * end * inc2;
                            var off3 = offset3 + t * end * inc3;
                            if (t == threads - 1) end = n - t * end;

                            for (int i = 0; i < end; i++)
                            {
                                x3[off3] = fn(x0[off0], x1[off1], x2[off2]);
                                off0 += inc0;
                                off1 += inc1;
                                off2 += inc2;
                                off3 += inc3;
                            }
                        });
                    }
                });
        }

        public static Array<R> Apply<T1, T2, T3, T4, R>(this Array<T1> a, Array<T2> b, Array<T3> c, Array<T4> d, Func<T1, T2, T3, T4, R> fn, Array<R> result = null)
        {
            return Array_.ElementwiseOp(a, b, c, d, result,
                (n, x0, offset0, inc0, x1, offset1, inc1, x2, offset2, inc2, x3, offset3, inc3, x4, offset4, inc4) =>
                {
                    if (n < MIN_SIZE_FOR_PARELLELISM)
                        for (int i = 0; i < n; i++)
                        {
                            x4[offset4] = fn(x0[offset0], x1[offset1], x2[offset2], x3[offset3]);
                            offset0 += inc0;
                            offset1 += inc1;
                            offset2 += inc2;
                            offset3 += inc3;
                            offset4 += inc4;
                        }
                    else
                    {
                        var threads = Blas.NThreads;
                        Parallel.For(0, threads, t =>
                        {
                            var end = (n - 1) / threads + 1;
                            var off0 = offset0 + t * end * inc0;
                            var off1 = offset1 + t * end * inc1;
                            var off2 = offset2 + t * end * inc2;
                            var off3 = offset3 + t * end * inc3;
                            var off4 = offset4 + t * end * inc4;
                            if (t == threads - 1) end = n - t * end;

                            for (int i = 0; i < end; i++)
                            {
                                x4[off4] = fn(x0[off0], x1[off1], x2[off2], x3[off3]);
                                off0 += inc0;
                                off1 += inc1;
                                off2 += inc2;
                                off3 += inc3;
                                off4 += inc4;
                            }
                        });
                    }
                });
        }

        public static Array<T> LoadText<T>(string path)
        {
            return Array(File.ReadLines(path).
                Select(line => line.Split(' ').Select(value => NumNet.Array<T>.Operators.Parse(value)).ToArray()).ToArray());
        }

        public static IEnumerable<int> GetOffsets<T>(this Array<T> t, int[] shape)
        {
            int dim = shape.Length;
            t.AssertOfDim(dim);

            if(dim == 1) // handle separatly the trivial vector case
            {
                yield return t.Offset;
                yield break;
            }

            var progress = new int[shape.Length];
            int maxCount = shape.Aggregate(1, (x, y) => x * y) / shape[dim - 1];
            for (int axis = 0; axis < dim - 1; ++axis)
                if (shape[axis] != t.Shape[axis] && t.Shape[axis] != 1)
                    t.AssertOfShape(shape); // throws RankException
            int count = 0;
            int off = t.Offset;
            while (count < maxCount)
            {
                yield return off;
                count += 1;
                int axis = dim - 2;
                progress[axis] += 1;
                if (t.Shape[axis] > 1) // if we aren't broadcasting
                    off += t.Stride[axis];

                while (axis >= 0 && progress[axis] == shape[axis])
                {
                    progress[axis] = 0;
                    if (t.Shape[axis] > 1) // if we aren't broadcasting
                        off += t.Stride[axis];
                    off -= t.Stride[axis + 1] * t.Shape[axis + 1];
                    axis -= 1;
                }
                //if (axis < 0) yield break;
            }
        }


        /// <summary>
        /// Fills the result array using the value from a, and the indexes from selected.
        /// </summary>
        /// <typeparam name="T">The type of a content</typeparam>
        /// <param name="thiz"></param>
        /// <param name="selected">the result of a ArgMax/ArgMin operation</param>
        /// <param name="axis"></param>
        /// <param name="axisSize"></param>
        /// <param name="keepDims"></param>
        /// <param name="result"></param>
        /// <returns></returns>
        public static Array<T> UnArgmax<T>(this Array<T> thiz, Array<int> selected, int axis, int axisSize, bool keepDims = false, Array<T> result = null)
        {
            thiz.AssertOfShape(selected);
            var dim = thiz.NDim + (keepDims ? 0 : 1);
            var shape = new int[dim];
            if (keepDims)
                System.Array.Copy(thiz.Shape, shape, dim);
            else
            {
                System.Array.Copy(thiz.Shape, 0, shape, 0, axis);
                System.Array.Copy(thiz.Shape, axis, shape, axis + 1, thiz.NDim - axis);
            }
            shape[axis] = axisSize;
            result = result ?? NN.Zeros<T>(shape);
            result.AssertOfShape(shape);
            var resultInc = result.Stride[axis];

            // HACK
            // as result have one more shape than thiz and selected, we have to lie about the number of shapes
            var resultSlices = new Slice[dim];
            for (int i = 0; i < dim; ++i)
                resultSlices[i] = Slicer._;
            if (!keepDims)
                resultSlices[axis] = 0;
            else
                resultSlices[axis] = Slicer.Until(1);
            var res = result[resultSlices];

            Array_.ElementwiseOp(thiz, selected, res,
                (n, x, offsetx, incx, s, offsetS, incS, r, offsetR, incR) =>
                {
                    for (int i = 0; i < n; ++i)
                    {
                        r[offsetR + resultInc * s[offsetS]] = x[offsetx];
                        offsetR += incR;
                        offsetS += incS;
                        offsetx += incx;
                    }
                });

            return result;
        }

        public static Array<T> Tile<T>(Array<T> a, params int[] reps)
        {
            // Construct an array by repeating A the number of times given by reps.

            // If `reps` has length ``d``, the result will have dimension of
            // ``max(d, A.ndim)``.

            // If ``A.ndim < d``, `A` is promoted to be d-dimensional by prepending new
            // axes. So a shape (3,) array is promoted to (1, 3) for 2-D replication,
            // or shape (1, 1, 3) for 3-D replication. If this is not the desired
            // behavior, promote `A` to d-dimensions manually before calling this
            // function.

            // If ``A.ndim > d``, `reps` is promoted to `A`.ndim by pre-pending 1's to it.
            // Thus for an `A` of shape (2, 3, 4, 5), a `reps` of (2, 2) is treated as
            // (1, 1, 2, 2).

            // Parameters
            // ----------
            // A : array_like
            //     The input array.
            // reps : array_like
            //     The number of repetitions of `A` along each axis.

            // Returns
            // -------
            // c : ndarray
            //     The tiled output array.

            // See Also
            // --------
            // repeat : Repeat elements of an array.

            // Examples
            // --------
            // >>> a = np.array([0, 1, 2])
            // >>> np.tile(a, 2)
            // array([0, 1, 2, 0, 1, 2])
            // >>> np.tile(a, (2, 2))
            // array([[0, 1, 2, 0, 1, 2],
            //        [0, 1, 2, 0, 1, 2]])
            // >>> np.tile(a, (2, 1, 2))
            // array([[[0, 1, 2, 0, 1, 2]],
            //        [[0, 1, 2, 0, 1, 2]]])

            // >>> b = np.array([[1, 2], [3, 4]])
            // >>> np.tile(b, 2)
            // array([[1, 2, 1, 2],
            //        [3, 4, 3, 4]])
            // >>> np.tile(b, (2, 1))
            // array([[1, 2],
            //        [3, 4],
            //        [1, 2],
            //        [3, 4]])
            var tup = reps;
            var d = tup.Length;
            //c = _nx.array(A, copy=False, subok=True, ndmin=d)
            var c = a;
            var shape = (int[])c.Shape.Clone();
            var oldLength = shape.Length;
            if (oldLength < d)
            {
                //System.Array.Resize(ref shape, d);
                //for (int i = oldLength; i < d; i++) shape[i] = 1;
                //c = c.Reshape(shape);
                var newShape = new int[d];
                for (int i = 0; i < d - oldLength; i++) newShape[i] = 1;
                System.Array.Copy(shape, 0, newShape, d - oldLength, oldLength);
                shape = newShape;
                c = c.Reshape(shape);
            }

            var n = Math.Max(c.Size, 1);
            if (d < c.NDim)
            {
                //tup = (1,)*(c.ndim - d) + tup
                throw new NotImplementedException();
            }
            for (int i = 0; i < tup.Length; i++)
            {
                var nrep = tup[i];
                if (nrep != 1)
                {
                    c = NN.Repeat(c.Reshape(-1, n), nrep, axis: 0);
                }
                var dim_in = shape[i];
                var dim_out = dim_in * nrep;
                shape[i] = dim_out;
                n /= Math.Max(dim_in, 1);
            }
            return c.Reshape(shape);
        }

        public static Array<T> Repeat<T>(Array<T> a, int n, int axis = 0)
        {
            // Q & D implementation....
            if (axis != 0) throw new NotImplementedException();
            var shape = (int[])a.Shape.Clone();
            shape[0] *= n;
            var result = new Array<T>(shape);
            for (int i = 0; i < a.Shape[0]; i++)
            {
                var row = a[i];
                for (int j = 0; j < n; j++)
                {
                    result[i * n + j] = row;
                }
            }
            return result;
        }

        public static Array<T> Switch<T>(Array<float> mask, Array<T> ifTrue, Array<T> ifFalse, Array<T> result = null) =>
            Apply(mask, ifTrue, ifFalse, (m, a, b) => (m > 0) ? a : b, result);

        public static Array<T> Switch<T>(Array<int> mask, Array<T> ifTrue, Array<T> ifFalse, Array<T> result = null) =>
            Apply(mask, ifTrue, ifFalse, (m, a, b) => (m > 0) ? a : b, result);

        public static Array<T> ShapePadLeft<T>(Array<T> a, int pad = 1)
        {
            if (pad == 0)
                return a;

            var shape = new int[pad + a.NDim];
            for (int i = 0; i < pad; ++i) shape[i] = 1;
            System.Array.Copy(a.Shape, 0, shape, pad, a.NDim);

            return a.Reshape(shape);
        }
    }
}
