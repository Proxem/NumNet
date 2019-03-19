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

using Proxem.BlasNet;
using Proxem.NumNet.Single;

using Real = System.Single;

namespace Proxem.NumNet
{
    using static ShapeUtil;

    public static partial class NN
    {
        public static Array<Real> Copy(Array<Real> a, Array<Real> result = null)
        {
            return a.Copy(result);
        }

        //private static Real FastExp(Real val)
        //{
        //    // http://citeseer.ist.psu.edu/schraudolph98fast.html
        //    // http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.57.1569
        //    long tmp = (long)(1512775 * val + (1072693248 - 60801));
        //    return BitConverter.Int64BitsToDouble(tmp << 32); // *(Real*)(&value);

        //    // TODO: Int64BitsToDouble like http://stackoverflow.com/questions/3874627/doubleing-point-comparison-functions-for-c-sharp
        //    // public static unsafe int FloatToInt32Bits( Real f )
        //    // {
        //    //    return *( (int*)&f );
        //    // }
        //}

        public static Array<Real> Exp(Array<Real> a, Array<Real> result = null)
        {
            return a.Apply(x => (Real)Math.Exp(x), result);
        }

        public static Array<Real> Log(Array<Real> a, Array<Real> result = null)
        {
            return a.Apply(x => (Real)Math.Log(x), result);
        }

        //public static Real FastPow(Real a, Real b)
        //{
        //    // http://martin.ankerl.com/2007/10/04/optimized-pow-approximation-for-java-and-c-c/
        //    long tmp = BitConverter.DoubleToInt64Bits(a);
        //    long tmp2 = (long)(b * (tmp - 4606921280493453312L)) + 4606921280493453312L;
        //    return BitConverter.Int64BitsToDouble(tmp2);
        //}

        public static Array<Real> Pow(Array<Real> a, Real b, Array<Real> result = null)
        {
            return a.Apply(x => (Real)Math.Pow(x, b), result);
        }

        public static Array<Real> Sq(Array<Real> a, Array<Real> result = null) => a.Apply(x => x * x, result: result);

        public static Array<Real> Tanh(Array<Real> a, Array<Real> result = null)
        {
            return a.Apply(x => (Real)Math.Tanh(x), result);
        }

        private static Real Sigmoid(Real x)
        {
            return (Real)(1 / (1 + Math.Exp(-x)));
        }

        public static Array<Real> Sigmoid(Array<Real> a, Array<Real> result = null)
        {
            return a.Apply(x => Sigmoid(x), result);
        }

        public static Array<Real> Rectify(Array<Real> a, Array<Real> result = null)
        {
            return a.Apply(x => x > 0 ? x : 0, result);
        }

        public static Array<Real> Sqrt(Array<Real> a, Array<Real> result = null)
        {
            return a.Apply(x => (Real)Math.Sqrt(x), result);
        }

        public static Array<Real> Abs(Array<Real> a, Array<Real> result = null)
        {
            return a.Apply(Math.Abs, result);
        }

        /// <summary>
        /// http://aelag.com/translation-of-theano-softmax-function
        /// </summary>
        /// <param name="a"></param>
        /// <param name="axis">The axis to compute the Softmax along, like in the Max function. Default value mimics Theano behavior</param>
        /// <param name="result"></param>
        /// <returns></returns>
        public static Array<Real> Softmax(Array<Real> a, int axis = -1, Array<Real> result = null, Array<Real> buffer = null)
        {
            var maxes = NN.Max(a, axis: axis, keepDims: true, result: buffer);
            var shifted = a.Sub(maxes, result: result);
            result = NN.Exp(shifted, result: result);
            var sum = NN.Sum(result, axis: axis, keepDims: true, result: maxes);
            //result = result.Apply(sum, (x, s) => Math.Max(x / s, 0.000001f), result: result);
            result = result.Div(sum, result: result);
            return result;
        }

        // "fast" softmax for 1D and 2D arrays
        public static Array<Real> Softmax_(Array<Real> a, Array<Real> result = null)
        {
            if (a.Shape.Length > 2) throw new RankException(string.Format("Must be 1-d or 2-d tensor, got {0}-d with shape ({1}).", a.Shape.Length, string.Join(", ", a.Shape)));
            if (result == null) result = Zeros<Real>(a.Shape);
            else result.AssertOfShape(a);

            if (a.Shape.Length == 1)
            {
                var max = a.Max();
                result = NN.Exp(a - max, result: result);
            }
            else
            {
                var maxes = NN.Zeros<Real>(a.Shape[0], 1);
                var vMax = maxes.Values;
                int off = a.Offset, offX;
                int incX = a.Stride[0], incY = a.Stride[1];
                int nX = a.Shape[0], nY = a.Shape[1];
                Real max = Real.NegativeInfinity;
                var v = a.Values;
                for (int i = 0; i < nX; ++i)
                {
                    offX = off;
                    max = Real.NegativeInfinity;
                    for (int j = 0; j < nY; ++j)
                    {
                        max = Math.Max(v[off], max);
                        off += incY;
                    }
                    off = offX + incX;
                    vMax[i] = max;
                }
                result = NN.Exp(a - maxes, result: result);
            }
            var sum = NN.Sum(result, axis: a.Shape.Length - 1, keepDims: true);
            result = result.Div(sum, result: result);
            //result = result.Apply(sum, (x, s) => Math.Max(x / s, 0.000001f), result: result);
            return result;
        }

        /// <summary>
        /// Numerically stable shortcut of Log(Sum(Exp(a), axis, keepsDim).
        /// </summary>
        public static Array<Real> LogSumExp(Array<Real> a, int axis = -1, bool keepDims = false, Array<Real> result = null)
        {
            if (axis < 0) axis += a.NDim;
            var b = NN.Max(a, axis: axis, keepDims: true);
            var sum = NN.Exp(a - b).Sum(axis: axis, keepDims: true, result: result);
            result = Apply(sum, b, (x, b_) => b_ + (Real)Math.Log(x), result: result);
            return keepDims ? result : result.Reshape(GetAggregatorResultShape(a, axis, keepDims));
        }

        public static Real Sum(Array<Real> a)
        {
            return a.Sum();
        }

        public static Array<Real> Sum(Array<Real> a, int axis, bool keepDims = false, Array<Real> result = null)
        {
            return a.Sum(axis, keepDims: keepDims, result: result);
        }

        public static Real Max(Array<Real> a)
        {
            return a.Max();
        }

        public static Array<Real> Max(Array<Real> a, int axis, bool keepDims = false, Array<Real> result = null)
        {
            return a.Max(axis, keepDims, result);
        }

        public static Real Min(Array<Real> a)
        {
            return a.Min();
        }

        public static Array<Real> Min(Array<Real> a, int axis)
        {
            return a.Min(axis);
        }

        public static Real Mean(Array<Real> a)
        {
            return a.Mean();
        }

        public static Array<Real> Mean(Array<Real> a, int axis, bool keepdims = false, Array<Real> result = null)
        {
            var s = Sum(a, axis, keepdims, result);
            //s.Mul(1f / a.Shape[axis], result: s);
            s.Scale((Real)1 / a.Shape[axis], result: s);
            return s;
        }

        public static Real Variance(Array<Real> a, Array<Real> mean = null)
        {
            mean = mean ?? Mean(a);
            return Mean(Sq(a - mean));
        }

        public static Array<Real> Variance(Array<Real> a, int axis, Array<Real> result = null)
        {
            int n = a.Shape[axis];
            var mean = Mean(a, axis, keepdims: true);
            var mean2 = Sq(mean, result: mean);
            var a2 = a.Sum(axis, x => x * x / (Real)n, result: result);
            return a2.Sub(mean2, result: a2);
        }

        public static int Argmax(Array<Real> a)
        {
            return a.Argmax();
        }

        public static Array<int> Argmax(Array<Real> a, int axis, Array<int> result = null)
        {
            return a.Argmax(axis, result: result);
        }

        public static int Argmin(Array<Real> a)
        {
            return a.Argmin();
        }

        public static Real Norm(Array<Real> a)
        {
            return (Real)Math.Sqrt(Norm2(a));
        }

        public static Real Norm2(Array<Real> a)
        {
            Real result = 0;
            Array_.ElementwiseOp(0, a, 0,
                (n, x, offsetx, incx) =>
                {
                    result += Blas.dot(n, x, offsetx, incx, x, offsetx, incx);
                });
            return result;
        }

        public static Array<Real> Norm(Array<Real> a, int axis, Array<Real> result =null)
        {
            return Norm2(a, axis, result).Map(x => (Real)Math.Sqrt(x));
        }

        public static Array<Real> Norm2(Array<Real> a, int axis, Array<Real> result = null)
        {
            if (axis < 0) axis = a.Shape.Length + axis;
            if (result == null)
                result = Zeros<Real>(GetAggregatorResultShape(a, axis, true));
            else if (result.NDim != a.NDim)
                result = result.Reshape(GetAggregatorResultShape(a, axis, true));

            Array_.ElementwiseOp(a, result, (n, x, offx, incx, y, offy, incy) =>
            {
                y[offy] = Blas.dot(n, x, offx, incx, x, offx, incx);
            }, axis);

            return result;
        }

        /// <summary>
        /// Implementations of Euclidean distance that doesn't create intermediary array (unlike NN.Norm(a - b)).
        /// Faster for small arrays, longer for really big arrays
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static Real EuclideanDistance(Array<Real> a, Array<Real> b)
        {
            return (Real)Math.Sqrt(EuclideanDistance2(a, b));
        }


        /// <summary>
        /// Implementations of squared Euclidean distance that doesn't create intermediary array (unlike NN.Norm(a - b)).
        /// Faster for small arrays, longer for really big arrays
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static Real EuclideanDistance2(Array<Real> a, Array<Real> b)
        {
            Real result = 0;
            Array_.ElementwiseOp(a, b,
                (n, x, offx, incx, y, offy, incy) =>
                {
                    for (int _ = 0; _ < n; ++_)
                    {
                        var d = x[offx] - y[offy];
                        result += d * d;
                        offx += incx;
                        offy += incy;
                    }
                });
            return result;
        }

        public static Real RowNorm(this Array<Real> a, int axis = 0)
        {
            axis = axis >= 0 ? axis : axis + a.NDim;
            return (Real)Math.Sqrt(Norm2(a) / a.Shape[axis]);
        }

        public static Array<Real> Sign(Array<Real> a, Array<Real> result = null)
        {
            return a.Map(x => x < 0f ? -1f : 1f, result);
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
        public static Array<Real> Dot(Array<Real> a, Array<Real> b, Array<Real> result = null, bool transA = false, bool transB = false, float alpha = 1, float beta = 0)
        {
            return a.Dot(b, result, alpha, beta, transA, transB);
        }

        /// <summary>
        /// Reshapes to 2 axes. The first <paramref name="dimensionsAsRows"/> axes of the array become the first axis of the returned value.
        /// The remaining ones form the second axis.
        /// </summary>
        /// <remarks>Ported from gnumpy.</remarks>
        private static Array<Real> Reshape2D(Array<Real> a, int dimensionsAsRows, bool allowCopy = true, bool forceCopy = false)
        {
            if (dimensionsAsRows < 0) dimensionsAsRows += a.NDim;
            var before = a.Shape.Take(dimensionsAsRows).Aggregate(1, (x, y) => x * y);
            var after = a.Shape.Skip(dimensionsAsRows).Aggregate(1, (x, y) => x * y);
            return a.Reshape(new[] { before, after }, allowCopy, forceCopy);
        }

        /// <summary>
        /// Dot the last axes of a with the first axes of b.
        /// </summary>
        /// Ported from gnumpy.
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="axes">The number of axes to sum</param>
        /// <returns></returns>
        public static Array<Real> TensorDot(this Array<Real> a, Array<Real> b, int axes)
        {
            return Reshape2D(a, a.NDim - axes).Dot(Reshape2D(b, axes))
                .Reshape(a.Shape.Take(a.NDim - axes).Concat(b.Shape.Skip(axes)).ToArray());
        }

        /// <summary>
        /// TensorDot, see http://deeplearning.net/software/theano/library/tensor/basic.html
        /// Given two tensors a and b,tensordot computes a generalized dot product over the provided axes. Theano’s implementation reduces all
        /// expressions to matrix or vector dot products and is based on code from
        /// <a href="http://www.cs.toronto.edu/~tijmen/gnumpy.html">Tijmen Tieleman’s gnumpy</a>.
        /// </summary>
        /// <remarks>
        /// The tensor dot can cause long copy of array if the axes of 'a' aren't the lasts, and axes of 'b' aren't the firsts.
        /// </remarks>
        public static Array<Real> TensorDot(Array<Real> a, IList<int> axesA, Array<Real> b, IList<int> axesB, Array<Real> result = null)
        {
            var removeA = axesA.Select(i => i < 0 ? i + a.NDim : i).ToArray();
            var removeB = axesB.Select(i => i < 0 ? i + b.NDim : i).ToArray();

            var n = removeA.Length;
            if (n != removeB.Length)
                throw new RankException(
                    $"The axes parameters of tensorDot should have the same size. Found [{string.Join(", ", axesA)}] and [{string.Join(", ", axesB)}]."
                );

            for (int d = 0; d < n; ++d)
                if (a.Shape[removeA[d]] != b.Shape[removeB[d]])
                    throw new RankException(
                        $"Can't do dot along axes of different shape: {nameof(a)}.Shape[{removeA[d]}] == {a.Shape[removeA[d]]} while {nameof(b)}.Shape[{removeB[d]}] == {b.Shape[removeB[d]]}"
                    );

            // Move the axes to sum over to the end of "a"
            var keptA = Enumerable.Range(0, a.NDim).Where(d => !removeA.Contains(d));
            var at = a.Transpose(keptA.Concat(removeA).ToArray());

            // Move the axes to sum over to the front of "b"
            var keptB = Enumerable.Range(0, b.NDim).Where(d => !removeB.Contains(d));
            var bt = b.Transpose(removeB.Concat(keptB).ToArray());

            var resultShape = keptA.Select(axis => a.Shape[axis]).Concat(keptB.Select(axis => b.Shape[axis])).ToArray();
            Array<Real> res;
            // if no result array is given, let Dot create it, else we need to reshape it.
            if (result != null)
            {
                result.AssertOfShape(resultShape);
                res = Reshape2D(at, a.NDim - n).Dot(Reshape2D(bt, n), result: Reshape2D(result, a.NDim - n));
            }
            else
            {
                var a2d = Reshape2D(at, a.NDim - n);
                var b2d = Reshape2D(bt, n);
                res = a2d.Dot(b2d);
            }

            return res.Reshape(resultShape);
        }

        public static Array<Real> DownSample_MaxPooling2d(Array<Real> arr, int pool_h, int pool_w, bool ignoreBorder = true)
        {
            int x_h = arr.Shape[0];
            int x_w = arr.Shape[1];
            int out_h = x_h / pool_h;
            int out_w = x_w / pool_w;

            if (ignoreBorder == false)
            {
                if (((x_h ^ pool_h) >= 0) && (x_h % pool_h != 0)) out_h++;
                if (((x_w ^ pool_w) >= 0) && (x_w % pool_w != 0)) out_w++;
            }

            int arr_y_max = 0;
            int arr_x_max = 0;

            var poolout = new Real[out_h, out_w];

            for (int y_out = 0; y_out < out_h; y_out++)
            {
                int y = y_out * pool_h;
                int y_min = y;
                int y_max = Math.Min(y + pool_h, x_h);
                for (int x_out = 0; x_out < out_w; x_out++)
                {
                    int x = x_out * pool_w;
                    int x_min = x;
                    int x_max = Math.Min(x + pool_w, x_w);
                    var value = Real.NegativeInfinity;
                    for (int arr_y = y_min; arr_y < y_max; arr_y++)
                    {
                        for (int arr_x = x_min; arr_x < x_max; arr_x++)
                        {
                            var new_value = arr.Item[arr_y, arr_x];
                            if (new_value > value)
                            {
                                value = new_value;
                                arr_y_max = arr_y;
                                arr_x_max = arr_x;
                            }
                        }
                    }
                    poolout[y_out, x_out] = value;
                }
            }

            return poolout;
        }
        public static Array<int> DownSample_MaxPooling2d_IndexArray(Array<Real> arr, int pool_h, int pool_w, bool ignoreBorder = true)
        {
            int x_h = arr.Shape[0];
            int x_w = arr.Shape[1];
            int out_h = x_h / pool_h;
            int out_w = x_w / pool_w;

            if (ignoreBorder == false)
            {
                if (((x_h ^ pool_h) >= 0) && (x_h % pool_h != 0)) out_h++;
                if (((x_w ^ pool_w) >= 0) && (x_w % pool_w != 0)) out_w++;
            }

            int arr_y_max = 0;
            int arr_x_max = 0;

            var switches = new int[out_h, out_w, 2];

            for (int y_out = 0; y_out < out_h; y_out++)
            {
                int y = y_out * pool_h;
                int y_min = y;
                int y_max = Math.Min(y + pool_h, x_h);
                for (int x_out = 0; x_out < out_w; x_out++)
                {
                    int x = x_out * pool_w;
                    int x_min = x;
                    int x_max = Math.Min(x + pool_w, x_w);
                    var value = Real.NegativeInfinity;
                    for (int arr_y = y_min; arr_y < y_max; arr_y++)
                    {
                        for (int arr_x = x_min; arr_x < x_max; arr_x++)
                        {
                            var new_value = arr.Item[arr_y, arr_x];
                            if (new_value > value)
                            {
                                value = new_value;
                                arr_y_max = arr_y;
                                arr_x_max = arr_x;
                            }
                        }
                    }
                    switches[y_out, x_out, 0] = arr_y_max;
                    switches[y_out, x_out, 1] = arr_x_max;
                }
            }

            return switches;
        }

        public static Array<Real> Unpooling(Array<Real> delta, Array<Real> arr, int pool_h, int pool_w, bool ignoreBorder = true)
        {

            var switches = DownSample_MaxPooling2d_IndexArray(arr, pool_h, pool_w, ignoreBorder: ignoreBorder);
            var unpooled = new Real[arr.Shape[0], arr.Shape[1]];

            for (int y = 0; y < delta.Shape[0]; y++)
            {
                for (int x = 0; x < delta.Shape[1]; x++)
                {
                    unpooled[(int)switches[y, x, 0], (int)switches[y, x, 1]] = (Real)delta[y, x];
                }
            }

            return unpooled;
        }

        public static Tuple<Array<Real>, Array<int>> NewDownSample_MaxPooling2d(Array<Real> arr, int pool_h, int pool_w, bool ignoreBorder = true)
        {
            int x_h = arr.Shape[0];
            int x_w = arr.Shape[1];
            int out_h = x_h / pool_h;
            int out_w = x_w / pool_w;

            if (ignoreBorder == false)
            {
                if (((x_h ^ pool_h) >= 0) && (x_h % pool_h != 0)) out_h++;
                if (((x_w ^ pool_w) >= 0) && (x_w % pool_w != 0)) out_w++;
            }

            int arr_y_max = 0;
            int arr_x_max = 0;

            var poolout = NN.Array(new Real[out_h, out_w]);
            var switches = NN.Array(new int[out_h, out_w, 2]);

            for (int y_out = 0; y_out < out_h; y_out++)
            {
                int y = y_out * pool_h;
                int y_min = y;
                int y_max = Math.Min(y + pool_h, x_h);
                for (int x_out = 0; x_out < out_w; x_out++)
                {
                    int x = x_out * pool_w;
                    int x_min = x;
                    int x_max = Math.Min(x + pool_w, x_w);
                    var value = Real.NegativeInfinity;
                    for (int arr_y = y_min; arr_y < y_max; arr_y++)
                    {
                        for (int arr_x = x_min; arr_x < x_max; arr_x++)
                        {
                            var new_value = arr.Item[arr_y, arr_x];
                            if (new_value > value)
                            {
                                value = new_value;
                                arr_y_max = arr_y;
                                arr_x_max = arr_x;
                            }
                        }
                    }
                    switches[y_out, x_out, 0] = arr_y_max;
                    switches[y_out, x_out, 1] = arr_x_max;
                    poolout[y_out, x_out] = value;
                }
            }

            return Tuple.Create(poolout, switches);
        }

        public static Array<Real> new_Unpooling(Array<Real> delta, Array<int> switches, int pool_h, int pool_w, bool ignoreBorder = true)
        {
            var sh = new int[2];
            var unpooled = NN.Array(new Real[switches.Shape[0] * pool_h, switches.Shape[1] * pool_w]);

            for (int y = 0; y < delta.Shape[0]; y++)
            {
                for (int x = 0; x < delta.Shape[1]; x++)
                {
                    unpooled[(int)switches[y, x, 0], (int)switches[y, x, 1]] = (Real)delta[y, x];
                }
            }

            return unpooled;
        }

        public static Array<Real> Cov(Array<Real> m, int ddof = 1, Array<Real> result = null)
        {
            // Estimate a covariance matrix, given data.

            // Covariance indicates the level to which two variables vary together.
            // If we examine N-dimensional samples, :math:`X = [x_1, x_2, ... x_N]^T`,
            // then the covariance matrix element :math:`C_{ij}` is the covariance of
            // :math:`x_i` and :math:`x_j`. The element :math:`C_{ii}` is the variance
            // of :math:`x_i`.

            // Parameters
            // ----------
            // m : array_like
            //     A 1-D or 2-D array containing multiple variables and observations.
            //     Each row of `m` represents a variable, and each column a single
            //     observation of all those variables. Also see `rowvar` below.
            // ddof : int, optional
            //     .. versionadded:: 1.5
            //     If not ``None`` normalization is by ``(N - ddof)``, where ``N`` is
            //     the number of observations; this overrides the value implied by
            //     ``bias``. The default value is ``None``.

            // Returns
            // -------
            // out : ndarray
            //     The covariance matrix of the variables.

            // See Also
            // --------
            // corrcoef : Normalized covariance matrix

            // Examples
            // --------
            // Consider two variables, :math:`x_0` and :math:`x_1`, which
            // correlate perfectly, but in opposite directions:

            // >>> x = np.array([[0, 2], [1, 1], [2, 0]]).T
            // >>> x
            // array([[0, 1, 2],
            //        [2, 1, 0]])

            // Note how :math:`x_0` increases while :math:`x_1` decreases. The covariance
            // matrix shows this clearly:

            // >>> np.cov(x)
            // array([[ 1., -1.],
            //        [-1.,  1.]])

            // Note that element :math:`C_{0,1}`, which shows the correlation between
            // :math:`x_0` and :math:`x_1`, is negative.

            // Further, note how `x` and `y` are combined:

            // >>> x = [-2.1, -1,  4.3]
            // >>> y = [3,  1.1,  0.12]
            // >>> X = np.vstack((x,y))
            // >>> print np.cov(X)
            // [[ 11.71        -4.286     ]
            //  [ -4.286        2.14413333]]
            // >>> print np.cov(x, y)
            // [[ 11.71        -4.286     ]
            //  [ -4.286        2.14413333]]
            // >>> print np.cov(x)
            // 11.71

            var fact = (Real)(m.Shape[1] - ddof);
            if (fact <= 0) throw new ArgumentException("Degrees of freedom <= 0 for slice");

            var X = m - NN.Mean(m, axis: 1, keepdims: true);
            //result = X.T.Dot(X, result: result);
            result = X.Dot(X.T, result: result);
            result.Scale((Real)1.0 / fact, result: result);
            return result;
        }

        /// <summary>
        /// The Ln of Dirichlet pdf for symetric alpha.
        /// </summary>
        /// <param name="alpha"></param>
        /// <param name="k"></param>
        /// <param name="ln_z_sum">Sum(Ln(z))</param>
        /// <returns>Ln(Dir_alpha(z))</returns>
        public static Real LnDirichlet(Real alpha, int k, Real ln_z_sum) => (Real)(ln_z_sum * (alpha - 1) - LnBeta(alpha, k));

        /// <summary>
        /// The Ln of Dirichlet pdf for symetric alpha.
        /// </summary>
        public static Real LnDirichletPdf(Real alpha, Array<Real> z)
        {
            z.AssertOfDim(1);
            return LnDirichlet(alpha, z.Shape[0], Log(z).Sum());
        }

        public static Real DirichletPdf(Real alpha, Array<Real> z) => (Real)Math.Exp(LnDirichletPdf(alpha, z));

        public static Array<Real> Maximum(Array<Real> a, Real m)
        {
            return a.Apply(_a => Math.Max(_a, m));
        }

        public static Array<Real> Minimum(Array<Real> a, Real m)
        {
            return a.Apply(_a => Math.Max(_a, m));
        }
    }
}
