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
using System.Globalization;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;

namespace Proxem.NumNet
{
    using static ShapeUtil;

    /// <summary>N-dimensionnal arrays. Inspired from <a href="http://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html">NumPy ndarray</a>.</summary>
    [DebuggerDisplay("{_debugString}")]
    public class Array<Type> : Strided<Type>, IEnumerable<Type>, IIndexable<Index, Type>
    {
        internal static Operators<Type> Operators;

        /// <summary>Underlying data</summary>
        public Type[] Values;

        static Array()
        {
            // There is a distinct static field Operators for each intantiated generic
            // In the code below, we set
            //      Operators<Double> Operators = new Double.Operators()
            //      Operators<Int32> Operators = new Int32.Operators()
            //      Operators<Single> Operators = new Single.Operators()
            // And more generally
            //      Operators<XX> Operators = new XX.Operators()
            // If XX.Operators does not exist, we fall back to the default implementation
            //
            // If only C# operator overloading could be implemented through extensions methods
            // like in https://smellegantcode.wordpress.com/2014/04/24/adventures-in-roslyn-adding-crazily-powerful-operator-overloading-to-c-6/
            // life would be much easier...
            var name = "Proxem.NumNet." + typeof(Type).Name + ".Operators";
            var type = System.Reflection.Assembly.GetExecutingAssembly().GetType(name, throwOnError: false);
            if (type != null)
            {
                Operators = (Operators<Type>)type.GetConstructor(EmptyArray<System.Type>.Value).Invoke(null);
            }
            else Operators = new Operators<Type>();
        }

        public Array(int[] shape, Type[] values, int offset, int[] stride)
        {
            this.Shape = shape;
            this.Values = values;
            this.Offset = offset;
            this.Stride = stride;
            Flags = (CheckTransposed() ? Flags.Transposed : 0) | (CheckContiguous() ? 0 : Flags.NotContiguous);
        }

        public Array(int[] shape, Type[] values) :
            this(shape, values, 0, StridedExtension.ComputeStride(shape))
        {
        }

        public Array(params int[] shape) :
            this(shape, new Type[StridedExtension.ComputeSize(shape)])
        {
        }

        private string _debugString
        {
            get
            {
                if (Size < 10)
                    return ToString();
                var shape = $"({string.Join(", ", Shape)})";
                var content = $"[{string.Join(", ", this.Take(10))}, ...]";
                return content + shape;
            }
        }

        private Array<Type> Transposed;

        /// <summary>The transposed of this Array. Reverse the axis.</summary>
        /// <remarks><c>[i, j, k] -> [k, j, i]</c></remarks>
        public Array<Type> T
        {
            get
            {
                if (this.Transposed == null)
                {
                    this.Transposed = this.Transpose();
                    this.Transposed.Transposed = this;
                }
                return this.Transposed;
            }
        }

        /// <summary>Implicit conversion from a Type to Array&lt;Type&gt;</summary>
        public static implicit operator Array<Type>(Type a)
        {
            return new Array<Type>(EmptyArray<int>.Value, new Type[] { a }, 0, EmptyArray<int>.Value);
        }

        /// <summary>Explicit conversion from Array to a scalar</summary>
        /// <exception cref="InvalidCastException"><c>if(a.Shape.Length != 1)</c></exception>
        public static explicit operator Type(Array<Type> a)
        {
            if (a.Shape.Length != 0) throw new InvalidCastException(string.Format("expected 0-length shape but got ({0})", string.Join(", ", a.Shape)));
            return a.Values[a.Offset];
        }

        /// <summary>Explicit conversion from Array to System.Array</summary>
        /// <exception cref="InvalidCastException"><c>if(a.Shape.Length != 1)</c></exception>
        public static explicit operator Type[](Array<Type> a)
        {
            if (a.Shape.Length != 1) throw new InvalidCastException(string.Format("expected 0-length shape but got ({0})", string.Join(", ", a.Shape)));
            var result = new Type[a.Shape[0]];
            if (a.Stride[0] != 1) throw new NotImplementedException();  // TODO
            Array.Copy(a.Values, a.Offset, result, 0, a.Shape[0]);
            return result;
        }

        /// <summary>Implicit conversion from System.Array to Array</summary>
        public static implicit operator Array<Type>(Type[] a)
        {
            return new Array<Type>(new int[] { a.Length }, a);
        }

        /// <summary>Implicit conversion from multidimensionnal array to Array</summary>
        public static implicit operator Array<Type>(Type[,] a)
        {
            int n = a.GetLength(0);
            int m = a.GetLength(1);
            var flatValues = new Type[n * m];
            Buffer.BlockCopy(a, 0, flatValues, 0, Buffer.ByteLength(flatValues));
            return new Array<Type>(new[] { n, m }, flatValues);
        }

        /// <summary>Implicit conversion from multidimensionnal array to Array</summary>
        public static implicit operator Array<Type>(Type[,,] a)
        {
            int n = a.GetLength(0);
            int m = a.GetLength(1);
            int l = a.GetLength(2);
            var flatValues = new Type[n * m * l];
            Buffer.BlockCopy(a, 0, flatValues, 0, Buffer.ByteLength(flatValues));
            return new Array<Type>(new[] { n, m, l }, flatValues);
        }

        /// <summary>Implicit conversion from multidimensionnal array to Array</summary>
        public static implicit operator Array<Type>(Type[,,,] a)
        {
            int n = a.GetLength(0);
            int m = a.GetLength(1);
            int l = a.GetLength(2);
            int o = a.GetLength(3);
            var flatValues = new Type[n * m * l * o];
            Buffer.BlockCopy(a, 0, flatValues, 0, Buffer.ByteLength(flatValues));
            return new Array<Type>(new[] { n, m, l, o }, flatValues);
        }

        /// <summary>Elementwise addition</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<Type> operator +(Array<Type> a, Array<Type> b) => Operators.Add(a, b);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<Type> operator +(Type a, Array<Type> b) => Operators.Add(a, b);
        /// <summary>Elementwise subtraction</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<Type> operator +(Array<Type> a, Type b) => Operators.Add(a, b);

        /// <summary>Elementwise negation</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<Type> operator -(Array<Type> a) => Operators.Neg(a);

        /// <summary>Elementwise subtraction</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<Type> operator -(Array<Type> a, Array<Type> b) => Operators.Sub(a, b);
        /// <summary>Elementwise subtraction</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<Type> operator -(Type a, Array<Type> b) => Operators.Sub(a, b);
        /// <summary>Elementwise subtraction</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<Type> operator -(Array<Type> a, Type b) => Operators.Sub(a, b);

        /// <summary>Elementwise multiplication</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<Type> operator *(Array<Type> a, Array<Type> b) => Operators.Mul(a, b);
        /// <summary>Multiplication with a scalar</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<Type> operator *(Array<Type> a, Type b) => Operators.Mul(b, a);
        /// <summary>Optimized multiplication with a scalar</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<Type> operator *(Type a, Array<Type> b) => Operators.Mul(a, b);

        /// <summary>Elementwise division</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<Type> operator /(Array<Type> a, Array<Type> b) => Operators.Div(a, b);
        /// <summary>Division with a scalar</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<Type> operator /(Array<Type> a, Type b) => Operators.Div(a, b);
        /// <summary>Division with a scalar</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<Type> operator /(Type a, Array<Type> b) => Operators.Div(a, b);

        /// <summary>Elementwise comparaison (1 for true, 0 for false)</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<Type> operator >(Array<Type> a, Array<Type> b) => Operators.Gt(a, b);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<Type> operator >(Array<Type> a, Type b) => Operators.Gt(a, b);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<Type> operator >(Type a, Array<Type> b) => Operators.Gt(a, b);

        /// <summary>Elementwise comparaison (1 for true, 0 for false)</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<Type> operator >=(Array<Type> a, Array<Type> b) => Operators.GtEq(a, b);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<Type> operator >=(Array<Type> a, Type b) => Operators.GtEq(a, b);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<Type> operator >=(Type a, Array<Type> b) => Operators.GtEq(a, b);

        /// <summary>Elementwise comparaison (1 for true, 0 for false)</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<Type> operator <(Array<Type> a, Array<Type> b) => b > a;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<Type> operator <(Array<Type> a, Type b) => b > a;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<Type> operator <(Type a, Array<Type> b) => b > a;

        /// <summary>Elementwise comparaison (1 for true, 0 for false)</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<Type> operator <=(Array<Type> a, Array<Type> b) => b >= a;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<Type> operator <=(Array<Type> a, Type b) => b >= a;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<Type> operator <=(Type a, Array<Type> b) => b >= a;

        /// <summary>Prints all the elements of the array into a string.</summary>
        public override string ToString()
        {
            if (this.Shape.Length == 0) return Convert.ToString(this.Values[this.Offset], CultureInfo.InvariantCulture);
            var result = new StringBuilder();
            ToString(result, 0, 0);
            return result.ToString();
        }

        private void ToString(StringBuilder sb, int offset, int axis)
        {
            if (axis == this.Shape.Length - 1)
            {
                sb.Append('[');
                for (int i = 0; i < this.Shape[axis]; i++)
                {
                    if (i != 0) sb.Append(' ');
                    sb.Append(Convert.ToString(this.Values[offset + this.Offset], CultureInfo.InvariantCulture));
                    offset += this.Stride[axis];
                }
                sb.Append(']');
            }
            else
            {
                sb.Append('[');
                for (int i = 0; i < this.Shape[axis]; i++)
                {
                    if (i != 0)
                    {
                        if (axis + 2 < this.Shape.Length) sb.Append('\n');
                        sb.Append(' ', axis + 1);
                    }
                    ToString(sb, offset, axis + 1);
                    if (i != this.Shape[axis] - 1)
                    {
                        sb.Append('\n');
                        offset += this.Stride[axis];
                    }
                }
                sb.Append(']');
            }
        }

        /// <summary> Setter for the content of this array. </summary>
        public Array<Type> _
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set
            {
                NN.Copy(value, this);
            }
        }

        /// <summary>A view on the Array that can only be indexed by coordinates (instead of slices)</summary>
        public IIndexable<Index, Type> Item
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get { return this; }
        }

        /// <summary>Access an item of the Array from its coordinates</summary>
        Type IIndexable<Index, Type>.this[Index i0]
        {
            get
            {
                if (1 != this.Shape.Length) throw new Exception("wrong number of indices");
                return this.Values[this.RavelIndices(i0)];
            }
            set
            {
                if (1 < this.Shape.Length)
                {
                    this[new[] { i0 }] = value;
                }
                else
                {
                    this.Values[this.RavelIndices(i0)] = value;
                }
            }
        }

        /// <summary>Access an item of the Array from its coordinates</summary>
        Type IIndexable<Index, Type>.this[Index i0, Index i1]
        {
            get
            {
                if (2 != this.Shape.Length) throw new Exception("wrong number of indices");
                return this.Values[this.RavelIndices(i0, i1)];
            }
            set
            {
                if (2 < this.Shape.Length)
                {
                    this[new[] { i0, i1 }] = value;
                }
                else
                {
                    this.Values[this.RavelIndices(i0, i1)] = value;
                }
            }
        }

        /// <summary>Access an item of the Array from its coordinates</summary>
        Type IIndexable<Index, Type>.this[Index i0, Index i1, Index i2]
        {
            get
            {
                if (3 != this.Shape.Length) throw new Exception("wrong number of indices");
                return this.Values[this.RavelIndices(i0, i1, i2)];
            }
            set
            {
                if (3 < this.Shape.Length)
                {
                    this[new[] { i0, i1, i2 }] = value;
                }
                else
                {
                    this.Values[this.RavelIndices(i0, i1, i2)] = value;
                }
            }
        }

        /// <summary>Access an item of the Array from its coordinates</summary>
        Type IIndexable<Index, Type>.this[params Index[] indices]
        {
            get
            {
                if (indices.Length != this.Shape.Length) throw new Exception("wrong number of indices");
                return this.Values[this.RavelIndices(indices)];
            }
            set
            {
                if (indices.Length < this.Shape.Length)
                {
                    // promote indices to slices
                    var slices = new Slice[this.Shape.Length];
                    for (int i = 0; i < slices.Length; i++)
                    {
                        slices[i] = i < indices.Length ? (Slice)indices[i] : ..;
                    }
                    this[slices] = new Array<Type>(new[] { 1 }, new Type[] { value }, 0, new int[] { 0 });
                    return;
                }
                this.Values[this.RavelIndices(indices)] = value;
            }
        }

        public FastArray<Type> FastArray
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                return new FastArray<Type> { Offset = this.Offset, Stride = this.Stride, Values = this.Values };
            }
        }

        /// <summary>Optimized access to a slice of an Array for singletons</summary>
        /// <example><code>
        /// var a = NN.Array(new []{
        ///     {0, 1, 2},
        ///     {3, 4, 5},
        ///     {6, 7, 8},
        /// });
        /// Console.WriteLine(a[0, 0]);                        // [0]
        /// Console.WriteLine(a[0]);                           // [0, 1, 2]
        /// </code></example>
        [IndexerName("Slice")]
        public Array<Type> this[params int[] singletons]
        {
            get
            {
                int count = this.Shape.Length - singletons.Length;
                if (count == 0)
                    return new Array<Type>(EmptyArray<int>.Value, this.Values, this.RavelIndices(singletons), EmptyArray<int>.Value);

                var shape = new int[count];
                var offset = this.Offset;
                var stride = new int[count];
                int i = 0;
                int j = 0;
                int k = 0;
                while (j < count || i < Shape.Length)
                {
                    var slice = k < singletons.Length ? (Slice)singletons[k] : ..;
                    var a = this.GetAbsoluteIndex(slice.Range.Start, i);
                    if (a < 0 || a >= Shape[i])
                        throw new ArgumentException($"Slice [{slice.ToString()}] isn't valid for axis {i} of dim {Shape[i]}");
                    offset += a * this.Stride[i];
                    if (!slice.IsSingleton())    // if singleton, skip axis
                    {
                        var b = this.GetAbsoluteIndex(slice.Stop, i);
                        if (slice.Step >= 0)
                        {
                            if (b < a || b > Shape[i])
                                throw new ArgumentException($"Slice [{slice.ToString()}] isn't valid for axis {i} of dim {Shape[i]}");
                        }
                        else
                        {
                            if (a < b || b >= Shape[i])
                                throw new ArgumentException($"Slice [{slice.ToString()}] isn't valid for axis {i} of dim {Shape[i]}");
                        }
                        shape[j] = (b - a) / slice.Step + ((b - a) % slice.Step == 0 ? 0 : 1);
                        stride[j] = this.Stride[i] * slice.Step;
                        ++j;
                    }
                    ++i;
                    ++k;
                }

                var result = new Array<Type>(shape, this.Values, offset, stride);
                result.Flags |= Flags.NotContiguous;        // TODO: refine => some case are still contiguous
                return result;
            }
            set
            {
                if (singletons.Length > this.Shape.Length) throw new RankException("too many indices");

                if (value.NDim == 0)
                {
                    var y = value.Values[value.Offset];
                    var offset = RavelIndices(singletons);
                    if (singletons.Length == NDim)
                        Values[offset] = y;
                    else
                        Array_.ElementwiseOp(singletons.Length, this, offset, (n, x, offsetx, incx) =>
                        {
                            for (int i = 0; i < n; i++)
                            {
                                x[offsetx] = y;
                                offsetx += incx;
                            }
                        });
                }
                else
                {
                    int sLength = singletons.Length;
                    int ndim = NDim;
                    int v_ndim = value.NDim;

                    for (int a = 0; a + sLength < NDim && a < v_ndim; ++a)
                        if (value.Shape[a] != 1 && Shape[a + sLength] != value.Shape[a])
                            throw new RankException($"Can't set slice of shape {AssertArray.FormatShape(this[singletons].Shape)} with values of a {AssertArray.FormatShape(value.Shape)} array.");

                    var lastAxis = this.Shape.Length - 1;
                    int offset = 0;
                    while (lastAxis != 0 && lastAxis < singletons.Length)
                    {
                        offset += this.GetAbsoluteIndex(singletons[lastAxis], lastAxis) * this.Stride[lastAxis];
                        --lastAxis;
                    }
                    Array_.ElementwiseOp(0, 0, lastAxis, this, offset, singletons, value, 0,
                    (n, x, offsetx, incx, y, offsety, incy) =>
                    {
                        //Blas.copy(n, b, offsetb, incb, a, offseta, inca);
                        if (incx == 1 && incy == 1)
                            Array.Copy(y, offsety, x, offsetx, n);
                        else
                            for (int i = 0; i < n; i++)
                            {
                                x[offsetx] = y[offsety];
                                offsetx += incx;
                                offsety += incy;
                            }
                    });
                }
            }
        }

        [IndexerName("Slice")]
        public Array<Type> this[params Index[] singletons]
        {
            get
            {
                int count = this.Shape.Length - singletons.Length;
                if (count == 0)
                    return new Array<Type>(EmptyArray<int>.Value, this.Values, this.RavelIndices(singletons), EmptyArray<int>.Value);

                var shape = new int[count];
                var offset = this.Offset;
                var stride = new int[count];
                int i = 0;
                int j = 0;
                int k = 0;
                while (j < count || i < Shape.Length)
                {
                    var slice = k < singletons.Length ? (Slice)singletons[k] : ..;
                    var a = this.GetAbsoluteIndex(slice.Range.Start, i);
                    if (a < 0 || a >= Shape[i])
                        throw new ArgumentException($"Slice [{slice.ToString()}] isn't valid for axis {i} of dim {Shape[i]}");
                    offset += a * this.Stride[i];
                    if (!slice.IsSingleton())    // if singleton, skip axis
                    {
                        var b = this.GetAbsoluteIndex(slice.Stop, i);
                        if (slice.Step >= 0)
                        {
                            if (b < a || b > Shape[i])
                                throw new ArgumentException($"Slice [{slice.ToString()}] isn't valid for axis {i} of dim {Shape[i]}");
                        }
                        else
                        {
                            if (a < b || b >= Shape[i])
                                throw new ArgumentException($"Slice [{slice.ToString()}] isn't valid for axis {i} of dim {Shape[i]}");
                        }
                        shape[j] = (b - a) / slice.Step + ((b - a) % slice.Step == 0 ? 0 : 1);
                        stride[j] = this.Stride[i] * slice.Step;
                        ++j;
                    }
                    ++i;
                    ++k;
                }

                var result = new Array<Type>(shape, this.Values, offset, stride);
                result.Flags |= Flags.NotContiguous;        // TODO: refine => some case are still contiguous
                return result;
            }
            set
            {
                if (singletons.Length > this.Shape.Length) throw new RankException("too many indices");

                if (value.NDim == 0)
                {
                    var y = value.Values[value.Offset];
                    var offset = RavelIndices(singletons);
                    if (singletons.Length == NDim)
                        Values[offset] = y;
                    else
                        Array_.ElementwiseOp(singletons.Length, this, offset, (n, x, offsetx, incx) =>
                        {
                            for (int i = 0; i < n; i++)
                            {
                                x[offsetx] = y;
                                offsetx += incx;
                            }
                        });
                }
                else
                {
                    int sLength = singletons.Length;
                    int ndim = NDim;
                    int v_ndim = value.NDim;

                    for (int a = 0; a + sLength < NDim && a < v_ndim; ++a)
                        if (value.Shape[a] != 1 && Shape[a + sLength] != value.Shape[a])
                            throw new RankException($"Can't set slice of shape {AssertArray.FormatShape(this[singletons].Shape)} with values of a {AssertArray.FormatShape(value.Shape)} array.");

                    var lastAxis = this.Shape.Length - 1;
                    int offset = 0;
                    while (lastAxis != 0 && lastAxis < singletons.Length)
                    {
                        offset += this.GetAbsoluteIndex(singletons[lastAxis], lastAxis) * this.Stride[lastAxis];
                        --lastAxis;
                    }
                    Array_.ElementwiseOp(0, 0, lastAxis, this, offset, singletons, value, 0,
                    (n, x, offsetx, incx, y, offsety, incy) =>
                    {
                        //Blas.copy(n, b, offsetb, incb, a, offseta, inca);
                        if (incx == 1 && incy == 1)
                            Array.Copy(y, offsety, x, offsetx, n);
                        else
                            for (int i = 0; i < n; i++)
                            {
                                x[offsetx] = y[offsety];
                                offsetx += incx;
                                offsety += incy;
                            }
                    });
                }
            }
        }

        /// <summary>Access to a slice of an Array</summary>
        /// <example><code>
        /// var a = NN.Array(new []{
        ///     {0, 1, 2},
        ///     {3, 4, 5},
        ///     {6, 7, 8},
        /// });
        /// Console.WriteLine(a[0, Slicer.From(1)]);           // [1, 2]
        /// Console.WriteLine(a[Slicer.Until(-1)]);            // [[0, 1, 2], [3, 4, 5]]
        /// Console.WriteLine(a[Slicer.Until(-1), Slicer._]);  // [[0, 1, 2], [3, 4, 5]]
        /// </code></example>
        [IndexerName("Slice")]
        public Array<Type> this[params Slice[] slices]
        {
            get
            {
                int count = this.Shape.Length;
                foreach (var slice in slices)
                {
                    if (slice.IsNewAxis()) ++count;
                    else if (slice.IsSingleton()) --count;       // Assert(slices[i].Step == 1);
                }
                if (count == 0)
                    return new Array<Type>(EmptyArray<int>.Value, this.Values, this.RavelIndicesStart(slices), EmptyArray<int>.Value);

                var shape = new int[count];
                var offset = this.Offset;
                var stride = new int[count];
                int i = 0;
                int j = 0;
                int k = 0;
                while (j < count || i < this.Shape.Length)
                {
                    var slice = k < slices.Length ? slices[k] : ..;
                    if (slice.IsNewAxis())
                    {
                        shape[j] = 1;
                        stride[j] = 0;
                        ++j;
                    }
                    else
                    {
                        var a = this.GetAbsoluteIndex(slice.Start, i);
                        if (a < 0 || a >= this.Shape[i]) throw new ArgumentException();
                        offset += a * this.Stride[i];
                        if (!slice.IsSingleton())    // if singleton, skip axis
                        {
                            var b = this.GetAbsoluteIndex(slice.Stop, i);
                            if (slice.Step >= 0)
                            {
                                if (b < a || b > this.Shape[i])
                                    throw new ArgumentException($"Can't slice axis {i} of shape {Shape[i]} with slice {slice}");
                            }
                            else
                            {
                                if (a < b || b >= this.Shape[i])
                                    throw new ArgumentException($"Can't slice axis {i} of shape {Shape[i]} with slice {slice}");
                            }
                            shape[j] = (b - a) / slice.Step + ((b - a) % slice.Step == 0 ? 0 : 1);
                            stride[j] = this.Stride[i] * slice.Step;
                            ++j;
                        }
                        ++i;
                    }
                    ++k;
                }

                return new Array<Type>(shape, this.Values, offset, stride);
            }
            set
            {
                if (slices.Length > this.Shape.Length) throw new RankException("too many indices");
                var lastAxis = this.Shape.Length - 1;
                int offset = 0;
                while (lastAxis != 0 && lastAxis < slices.Length && slices[lastAxis].IsSingleton())
                {
                    offset += this.GetAbsoluteIndex(slices[lastAxis].Start, lastAxis) * this.Stride[lastAxis];
                    --lastAxis;
                }
                Array_.ElementwiseOp(0, 0, lastAxis, this, offset, slices, value, 0,
                    (n, x, offsetx, incx, y, offsety, incy) =>
                    {
                        //Blas.copy(n, b, offsetb, incb, a, offseta, inca);
                        if (incx == 1 && incy == 1)
                            Array.Copy(y, offsety, x, offsetx, n);
                        else
                            for (int i = 0; i < n; i++)
                            {
                                x[offsetx] = y[offsety];
                                offsetx += incx;
                                offsety += incy;
                            }
                    });
            }
        }

        /// <summary>Reads the array with <a href="http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#advanced-indexing">advanced-indexing</a></summary>
        /// <remarks>
        /// Generally speaking, what is returned when index arrays are used is an array with the same shape as the index array,
        /// but with the type and values of the array being indexed.
        /// Things become more complex when multidimensional arrays are indexed, particularly with multidimensional index arrays.
        /// In this case, if the index arrays have a matching shape, and there is an index array for each dimension of the array
        /// being indexed, the resultant array has the same shape as the index arrays, and the values correspond to the index set
        /// for each position in the index arrays.
        /// If the index arrays do not have the same shape, there is an attempt to broadcast them to the same shape. If they cannot
        /// be broadcast to the same shape, an exception is raised
        /// </remarks>
        /// <example><code>
        /// var a = NN.Array(new []{
        ///     {0, 1, 2},
        ///     {3, 4, 5},
        ///     {6, 7, 8},
        /// });
        /// Console.WriteLine(a[NN.Array(0, 2)]);                  // [[0, 1, 2], [6, 7, 8]]
        /// Console.WriteLine(a[NN.Array(0, 2), NN.Array(1, 0)]);  // [1, 6]
        /// </code></example>
        [IndexerName("Slice")]
        public Array<Type> this[params Array<int>[] indexArrays] => IndexWith(indexArrays);

        [IndexerName("Slice")]
        public Array<Type> this[params Array<Index>[] indexArrays] => IndexWith(indexArrays);

        /// <summary> Advanced indexing </summary>
        public Array<Type> IndexWith(Array<int>[] indexArrays, Array<Type> result = null)
        {
            // http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#advanced-indexing
            if (indexArrays.Length <= this.Shape.Length)
            {
                // "When the index consists of as many integer arrays as the array being indexed has dimensions, the indexing is"
                // "straight forward, but different from slicing."

                // "Note that the result shape is identical to the (broadcast) indexing array shapes ind_1, ..., ind_N."
                var firstArray = indexArrays[0];
                var shape = firstArray.Shape;
                var ndim = shape.Length;
                for (int i = 1; i < indexArrays.Length; i++)
                {
                    // TODO: broadcast
                    if (indexArrays[i].NDim != ndim)
                        throw new NotImplementedException("Arrays don't have same shapes");
                    for (int d = 0; d < ndim; ++d)
                        if (indexArrays[i].Shape[d] != shape[d])
                            throw new NotImplementedException("Arrays don't have same shapes");
                }

                var extra = this.Shape.Length - indexArrays.Length;
                if (extra != 0)
                {
                    shape = new int[firstArray.Shape.Length + extra];
                    Array.Copy(firstArray.Shape, shape, firstArray.Shape.Length);
                    Array.Copy(this.Shape, this.Shape.Length - extra, shape, firstArray.Shape.Length, extra);
                }

                // "Advanced indexing always returns a copy of the data (contrast with basic slicing that returns a view)."
                result?.AssertOfShape(shape);
                result = result ?? new Array<Type>(shape);
                if (firstArray.NDim == 0)
                {
                    result = this.Item[(int)indexArrays[0]];
                    return result;
                }
                else if (firstArray.Shape.Length == 1)
                {
                    var indices = new Index[indexArrays.Length];
                    for (int axis0 = 0; axis0 < firstArray.Shape[0]; axis0++)
                    {
                        for (int i = 0; i < indices.Length; i++)
                        {
                            var index = indexArrays[i].Item[axis0];
                            indices[i] = index >= 0 ? index : new Index(-index, fromEnd: true);
                        }
                        if (result.NDim == 1)
                            result.Item[axis0] = this.Item[indices];
                        else
                            result[axis0] = this[indices];
                    }
                    return result;
                }
                else if (firstArray.Shape.Length == 2)
                {
                    var indices = new Index[indexArrays.Length];
                    for (int axis0 = 0; axis0 < firstArray.Shape[0]; axis0++)
                        for (int axis1 = 0; axis1 < firstArray.Shape[1]; axis1++)
                        {
                            for (int i = 0; i < indices.Length; i++)
                                indices[i] = indexArrays[i].Item[axis0, axis1];
                            if (result.NDim == 2)
                                result.Item[axis0, axis1] = this.Item[indices];
                            else
                                result[axis0, axis1] = this[indices];
                        }
                    return result;
                }
                else if (firstArray.Shape.Length == 3)
                {
                    var fastResult = result.FastArray;
                    var indices = new int[indexArrays.Length];
                    for (int axis0 = 0; axis0 < firstArray.Shape[0]; ++axis0)
                        for (int axis1 = 0; axis1 < firstArray.Shape[1]; ++axis1)
                            for (int axis2 = 0; axis2 < firstArray.Shape[2]; ++axis2)
                            {
                                if (result.NDim == 3)
                                {
                                    var offset = this.Offset;
                                    for (int i = 0; i < indices.Length; ++i)
                                    {
                                        //indices[i] = indexArrays[i].Item[axis0, axis1, axis2];
                                        //indices[i] = indexArrays[i].FastArray[axis0, axis1, axis2];
                                        var ia = indexArrays[i];
                                        var stride = ia.Stride;
                                        //indices[i] = ia.Values[ia.Offset + axis0 * stride[0] + axis1 * stride[1] + axis2 * stride[2]];
                                        var index = ia.Values[ia.Offset + axis0 * stride[0] + axis1 * stride[1] + axis2 * stride[2]];
                                        if (index < 0) index += this.Shape[i];
                                        offset += this.Stride[i] * index;
                                    }
                                    //result.Item[axis0, axis1, axis2] = this.Item[indices];
                                    //fastResult[axis0, axis1, axis2] = ;
                                    //var v1 = this.Item[indices];
                                    var v2 = this.Values[offset];
                                    fastResult[axis0, axis1, axis2] = v2;
                                }
                                else
                                {
                                    for (int i = 0; i < indices.Length; ++i)
                                    {
                                        //indices[i] = indexArrays[i].Item[axis0, axis1, axis2];
                                        //indices[i] = indexArrays[i].FastArray[axis0, axis1, axis2];
                                        var ia = indexArrays[i];
                                        var stride = ia.Stride;
                                        indices[i] = ia.Values[ia.Offset + axis0 * stride[0] + axis1 * stride[1] + axis2 * stride[2]];
                                    }
                                    result[axis0, axis1, axis2] = this[indices];
                                }
                            }
                    return result;
                }
                else
                    // TODO
                    throw new NotImplementedException("Indexing with arrays of ndim > 3 is not supported yet");
            }
            else
                throw new ArgumentException($"A {NDim} array can't be indexed with {indexArrays.Length} arrays (max is {NDim}).");
        }

        /// <summary> Advanced indexing </summary>
        public Array<Type> IndexWith(Array<Index>[] indexArrays, Array<Type> result = null)
        {
            // http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#advanced-indexing
            if (indexArrays.Length <= this.Shape.Length)
            {
                // "When the index consists of as many integer arrays as the array being indexed has dimensions, the indexing is"
                // "straight forward, but different from slicing."

                // "Note that the result shape is identical to the (broadcast) indexing array shapes ind_1, ..., ind_N."
                var firstArray = indexArrays[0];
                var shape = firstArray.Shape;
                var ndim = shape.Length;
                for (int i = 1; i < indexArrays.Length; i++)
                {
                    // TODO: broadcast
                    if (indexArrays[i].NDim != ndim)
                        throw new NotImplementedException("Arrays don't have same shapes");
                    for (int d = 0; d < ndim; ++d)
                        if (indexArrays[i].Shape[d] != shape[d])
                            throw new NotImplementedException("Arrays don't have same shapes");
                }

                var extra = this.Shape.Length - indexArrays.Length;
                if (extra != 0)
                {
                    shape = new int[firstArray.Shape.Length + extra];
                    Array.Copy(firstArray.Shape, shape, firstArray.Shape.Length);
                    Array.Copy(this.Shape, this.Shape.Length - extra, shape, firstArray.Shape.Length, extra);
                }

                // "Advanced indexing always returns a copy of the data (contrast with basic slicing that returns a view)."
                result?.AssertOfShape(shape);
                result = result ?? new Array<Type>(shape);
                if (firstArray.NDim == 0)
                {
                    result = this.Item[(Index)indexArrays[0]];
                    return result;
                }
                else if (firstArray.Shape.Length == 1)
                {
                    var indices = new Index[indexArrays.Length];
                    for (int axis0 = 0; axis0 < firstArray.Shape[0]; axis0++)
                    {
                        for (int i = 0; i < indices.Length; i++)
                        {
                            var index = indexArrays[i].Item[axis0];
                            indices[i] = index;
                        }
                        if (result.NDim == 1)
                            result.Item[axis0] = this.Item[indices];
                        else
                            result[axis0] = this[indices];
                    }
                    return result;
                }
                else if (firstArray.Shape.Length == 2)
                {
                    var indices = new Index[indexArrays.Length];
                    for (int axis0 = 0; axis0 < firstArray.Shape[0]; axis0++)
                        for (int axis1 = 0; axis1 < firstArray.Shape[1]; axis1++)
                        {
                            for (int i = 0; i < indices.Length; i++)
                                indices[i] = indexArrays[i].Item[axis0, axis1];
                            if (result.NDim == 2)
                                result.Item[axis0, axis1] = this.Item[indices];
                            else
                                result[axis0, axis1] = this[indices];
                        }
                    return result;
                }
                else if (firstArray.Shape.Length == 3)
                {
                    var fastResult = result.FastArray;
                    var indices = new Index[indexArrays.Length];
                    for (int axis0 = 0; axis0 < firstArray.Shape[0]; ++axis0)
                        for (int axis1 = 0; axis1 < firstArray.Shape[1]; ++axis1)
                            for (int axis2 = 0; axis2 < firstArray.Shape[2]; ++axis2)
                            {
                                if (result.NDim == 3)
                                {
                                    var offset = this.Offset;
                                    for (int i = 0; i < indices.Length; ++i)
                                    {
                                        //indices[i] = indexArrays[i].Item[axis0, axis1, axis2];
                                        //indices[i] = indexArrays[i].FastArray[axis0, axis1, axis2];
                                        var ia = indexArrays[i];
                                        var stride = ia.Stride;
                                        //indices[i] = ia.Values[ia.Offset + axis0 * stride[0] + axis1 * stride[1] + axis2 * stride[2]];
                                        var index = ia.Values[ia.Offset + axis0 * stride[0] + axis1 * stride[1] + axis2 * stride[2]];
                                        if (index.IsFromEnd)
                                            offset += this.Stride[i] * (index.Value + this.Shape[i]);
                                        else 
                                            offset += this.Stride[i] * index.Value;
                                    }
                                    //result.Item[axis0, axis1, axis2] = this.Item[indices];
                                    //fastResult[axis0, axis1, axis2] = ;
                                    //var v1 = this.Item[indices];
                                    var v2 = this.Values[offset];
                                    fastResult[axis0, axis1, axis2] = v2;
                                }
                                else
                                {
                                    for (int i = 0; i < indices.Length; ++i)
                                    {
                                        //indices[i] = indexArrays[i].Item[axis0, axis1, axis2];
                                        //indices[i] = indexArrays[i].FastArray[axis0, axis1, axis2];
                                        var ia = indexArrays[i];
                                        var stride = ia.Stride;
                                        indices[i] = ia.Values[ia.Offset + axis0 * stride[0] + axis1 * stride[1] + axis2 * stride[2]];
                                    }
                                    result[axis0, axis1, axis2] = this[indices];
                                }
                            }
                    return result;
                }
                else
                    // TODO
                    throw new NotImplementedException("Indexing with arrays of ndim > 3 is not supported yet");
            }
            else
                throw new ArgumentException($"A {NDim} array can't be indexed with {indexArrays.Length} arrays (max is {NDim}).");
        }

        public IEnumerator<Type> GetEnumerator()
        {
            if (Shape.Length == 0) yield break;

            int lastAxis = this.Shape.Length - 1;
            if (lastAxis == -1)
            {
                yield return this.Values[this.Offset];
                yield break;
            }
            int count = this.Size;

            var coord = new int[this.Shape.Length];
            coord[lastAxis] -= 1;
            var off = this.Offset - this.Stride[lastAxis];

            for (var step = 0; step < count; step++)
            {
                ++coord[lastAxis];
                off += this.Stride[lastAxis];

                var axis = lastAxis;
                while (axis > 0 && coord[axis] == this.Shape[axis])
                {
                    coord[axis] = 0;
                    off -= this.Shape[axis] * this.Stride[axis];

                    --axis;
                    ++coord[axis];
                    off += this.Stride[axis];
                }
                yield return this.Values[off];
            }
        }

        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public Array<Type> MinifyDim()
        {
            if (this.Shape.Length == 1)
                return this;

            var lastAxis = this.Shape.Length - 1;
            while (lastAxis >= 0 && this.Shape[lastAxis] == 1) --lastAxis;
            var firstAxis = 0;
            while (firstAxis < lastAxis && this.Shape[firstAxis] == 1) ++firstAxis;
            if (lastAxis == -1)
                return this;
            else if (firstAxis == 0 && lastAxis == this.Shape.Length - 1)
                return this;
            else
            {
                var shape = new int[lastAxis - firstAxis + 1];
                Array.Copy(this.Shape, firstAxis, shape, 0, lastAxis - firstAxis + 1);
                return this.Reshape(shape);
            }
        }

        public bool Equals(Array<Type> that)
        {
            if (this.Size != that.Size)
                return false;
            else
                return ElementwiseCheck(0, this.MinifyDim(), 0, that.MinifyDim(), 0,
                    (n, x, offsetx, incx, y, offsety, incy) =>
                    {
                        int ja = offsetx, jb = offsety;
                        for (int i = 0; i < n; i++)
                        {
                            if (!x[ja].Equals(y[jb]))
                                return false;
                            ja += incx;
                            jb += incy;
                        }
                        return true;
                    });
        }

        public Array<Type> Reshape(params int[] shape)
        {
            return Reshape(shape, allowCopy: true, forceCopy: false);
        }

        /// <summary>
        /// Reshapes the array to the given shape.
        /// The reshape operation tries to be a O(1) and to reuse the underlying values.
        /// But if the data is not contiguous (for instance if this array is transposed), the reshape need to copy the values.
        /// (Note: like in NumPy the default is allowCopy=true).
        /// If one of the <paramref name="shape"/> value is -1, will try to guess the correct value.
        /// </summary>
        /// <param name="allowCopy">Allows to copy the values. If false Reshape will throw an ArgumentException if a copy is needed.</param>
        /// <param name="forceCopy">Force the copy. Will copy the values in a new Array.</param>
        /// <exception cref="RankException">if the new shape doesn't match with old one.</exception>
        /// <exception cref="ArgumentException">if the Reshape require a copy and <paramref name="allowCopy"/> is false.</exception>
        public Array<Type> Reshape(int[] shape, bool allowCopy = true, bool forceCopy = false)
        {
            var oldsize = this.Size;
            var newsize = 1;
            var negpos = -1;
            bool sameShape = shape.Length == this.NDim;
            for (int i = 0; i < shape.Length; i++)
            {
                if (shape[i] == -1)
                {
                    if (negpos != -1)
                        throw new ArgumentException($"Only one axis can have a dim of -1 in [{string.Join(", ", shape)}]");
                    negpos = i;
                }
                else
                    newsize *= shape[i];
                if (sameShape && shape[i] != Shape[i])
                    sameShape = false;
            }
            if (negpos != -1)
            {
                shape[negpos] = oldsize / newsize;
                newsize = newsize * shape[negpos];
            }
            if (newsize != oldsize)
                throw new RankException($"Can't reshape from size {oldsize} to size {newsize}");

            if (sameShape && !forceCopy)
                return this;

            // if IsTransposed() we are forced to make a copy.
            var needCopy = IsTransposed();

            if (forceCopy || (needCopy && allowCopy))
            {
                if(IsContiguous() && Stride[NDim - 1] == 1)
                {
                    var copiedValues = new Type[newsize];
                    Array.Copy(this.Values, this.Offset, copiedValues, 0, newsize);
                    return new Array<Type>(shape, copiedValues);
                }
                else
                {
                    // TODO: optimize (use Copy ??)
                    return new Array<Type>(shape, this.AsEnumerable().ToArray());
                }
            }
            else if (!needCopy)
            {
                var stride = new int[shape.Length];
                if(stride.Length > 0 && this.NDim > 0)
                    stride[stride.Length - 1] = this.Stride[this.NDim - 1];
                for (int i = stride.Length - 2; i >= 0; i--)
                    stride[i] = stride[i + 1] * shape[i + 1];

                return new Array<Type>(shape, this.Values, this.Offset, stride);
            }
            else
                throw new ArgumentException("Can't reshape like this without copying.");
        }

        /// <see cref="T">T</see>
        public Array<Type> Transpose()
        {
            int count = this.Shape.Length;
            //if (count == 1) return TransposeVector();

            var shape = new int[count];
            var stride = new int[count];
            for (int i = 0; i < count; i++)
            {
                shape[i] = this.Shape[count - i - 1];
                stride[i] = this.Stride[count - i - 1];
            }
            var result = new Array<Type>(shape, this.Values, this.Offset, stride);
            if ((this.Flags & Flags.Transposed) == 0) result.Flags |= Flags.Transposed;
            return result;
        }

        // Useful ? see Dot(..., bool transA = false, bool transB = false)
        private Array<Type> TransposeVector()
        {
            int vecSize = this.Shape[0];
            var shape = new int[] { 1, vecSize };
            var stride = new int[] { 0, this.Stride[0] };
            var result = new Array<Type>(shape, this.Values, this.Offset, stride);
            if (!this.IsTransposed()) result.Flags |= Flags.Transposed;
            return result;
        }

        public Array<Type> Transpose(params int[] axesPerm)
        {
            int count = NDim;
            if (count != axesPerm.Length) throw new ArgumentException();
            if (IsIdentityPerm(axesPerm))
                return this;

            var shape = new int[count];
            var stride = new int[count];
            for (int i = 0; i < count; i++)
            {
                shape[i] = this.Shape[axesPerm[i]];
                stride[i] = this.Stride[axesPerm[i]];
            }
            return new Array<Type>(shape, Values, Offset, stride);
        }

        private bool IsIdentityPerm(int[] axesPerm)
        {
            for (int i = 0; i < axesPerm.Length; ++i)
                if (axesPerm[i] != i)
                    return false;
            return true;
        }

        /// <summary>Casts all the elements of the array to the given type</summary>
        public Array<T> As<T>(Array<T> result = null)
        {
            result?.AssertOfShape(this.Shape);
            result = result ?? new Array<T>(this.Shape);
            Array_.ElementwiseOp(this, result, (n, x, offsetx, incx, y, offsety, incy) =>
            {
                var type = typeof(T);
                for (int i = 0; i < n; i++)
                {
                    y[offsety] = (T)Convert.ChangeType(x[offsetx], type);
                    offsetx += incx;
                    offsety += incy;
                }
            });
            return result;
        }

        /// <summary>Creates an empty array with the same shape</summary>
        /// <typeparam name="R">The type of the new Array</typeparam>
        public Array<R> Empty<R>()
        {
            return new Array<R>(this.Shape);
        }

        /// <summary>Creates an empty array with the same shape and same type</summary>
        public Array<Type> Empty()
        {
            return new Array<Type>(this.Shape);
        }

        /// <summary>Fills this array with the given values</summary>
        public void FillWith(IEnumerable<Type> input)
        {
            FillWith(input.GetEnumerator());
        }

        /// <summary>Fills this array with the given values</summary>
        public void FillWith(IEnumerator<Type> it)
        {
            Array_.ElementwiseOp(this, StridedUtil.StoreResult(() => { it.MoveNext(); return it.Current; }));
        }

        public void FillWith(Type value) => Array_.ElementwiseOp(this, (n, r, offR, strideR) =>
        {
            for (int i = 0; i < n; ++i)
            {
                r[offR] = value;
                offR += strideR;
            }
        });

        public Array<Type> Insert(Array<Type> other, int index, int axis, Array<Type> result = null)
        {
            if (other.Shape.Length != this.Shape.Length - 1) throw new RankException();
            index = GetAbsoluteIndex(index, axis);
            int j = 0;
            var newShape = new int[this.Shape.Length];
            for (int i = 0; i < this.Shape.Length; i++)
            {
                if (i == axis)
                {
                    newShape[i] = this.Shape[i] + 1;
                }
                else
                {
                    if (other.Shape[j++] != this.Shape[i]) throw new RankException();
                    newShape[i] = this.Shape[i];
                }
            }
            if (result != null) result.AssertOfShape(newShape);
            else result = new Array<Type>(newShape);

            if (axis != 0) throw new NotImplementedException();     // TODO
            for (int i = 0; i < newShape[axis]; i++)
            {
                if (i < index) result[i] = this[i];
                else if (i == index) result[i] = other;
                else result[i] = this[i - 1];
            }
            return result;
        }

        #region Elementwise operations

        public Array<R> Map<R>(Func<Type, R> f, Array<R> result = null)
        {
            if (result == null) result = NN.Zeros<R>(this.Shape);
            else result.AssertOfShape(Shape);
            var op = StridedUtil.StoreResult(f);
            Array_.ElementwiseOp(this, result, op);
            return result;
        }

        public Array<R> Zip<T, R>(Array<T> that, Func<Type, T, R> f, Array<R> result)
        {
            var op = StridedUtil.StoreResult(f);
            Array_.ElementwiseOp(0, this, 0, that, 0, result, 0, op);
            return result;
        }

        public Array<Type> ZipInPlace<T1>(Array<T1> that, Func<Type, T1, Type> f)
        {
            var op = StridedUtil.ComputeAndStore(f);
            Array_.ElementwiseOp(this, that, op);
            return this;
        }

        public Array<Type> ZipInPlace<T1, T2>(Array<T1> that, Array<T2> andThat, Func<Type, T1, T2, Type> f)
        {
            var op = StridedUtil.ComputeAndStore(f);
            // ElementwiseOp requires that andThat.Shape == BroadcastShapes(this, that)
            if (!CheckShapes(this, that, andThat))
            {
                andThat.Reshape(BroadcastShapes(this, that));
            }
            Array_.ElementwiseOp(0, this, 0, that, 0, andThat, 0, op);
            return this;
        }

        internal static bool ElementwiseCheck(int axis,
            Array<Type> a, int offseta,
            Func<int, Type[], int, int, bool> op)
        {
            var lastAxis = a.Shape.Length - 1;
            while (lastAxis != 0 && a.Shape[lastAxis] == 1) --lastAxis;
            if (axis == lastAxis)
            {
                return op(a.Shape[axis], a.Values, offseta + a.Offset, a.Stride[axis]);
            }
            else
            {
                for (int i = 0; i < a.Shape[axis]; i++)
                {
                    if (!ElementwiseCheck(axis + 1, a, offseta, op))
                        return false;
                    offseta += a.Stride[axis];
                }
                return true;
            }
        }

        internal static bool ElementwiseCheck<T2>(int axis,
            Array<Type> a, int offseta,
            Array<T2> b, int offsetb,
            Func<int, Type[], int, int, T2[], int, int, bool> op)
        {
            var lastAxis = a.Shape.Length - 1;
            while (lastAxis != 0 && a.Shape[lastAxis] == 1 && b.Shape[lastAxis] == 1)
            {
                --lastAxis;
            }
            if (axis == lastAxis)
            {
                return op(a.Shape[axis], a.Values, offseta + a.Offset, a.Stride[axis], b.Values, offsetb + b.Offset, b.Stride[axis]);
            }
            else
            {
                for (int i = 0; i < a.Shape[axis]; i++)
                {
                    if (!ElementwiseCheck(axis + 1, a, offseta, b, offsetb, op))
                        return false;
                    offseta += a.Stride[axis];
                    offsetb += b.Stride[axis];
                }
                return true;
            }
        }
        #endregion

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool IsContiguous() { return (this.Flags & Flags.NotContiguous) == 0; }

        private bool CheckContiguous()
        {
            var contiguous = true;
            for (int d = 0; d < NDim - 1; ++d)
                if (Stride[d] != Shape[d + 1] * Stride[d + 1])
                    contiguous = false;
            return contiguous;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool IsTransposed() { return (this.Flags & Flags.Transposed) != 0; }

        private bool CheckTransposed()
        {
            var transposed = false;
            for (int d = 0; d < NDim - 1; ++d)
                if (Stride[d] < Stride[d + 1])
                    transposed = true;
            return transposed;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int LastStride() { return Stride[Stride.Length - 1]; }
    }

    [Flags]
    public enum Flags
    {
        Transposed = 0x1,
        NotContiguous = 0x2
    }

    public interface IIndexable<T, Type>
    {
        Type this[T i0] { get; set; }
        Type this[T i0, T i1] { get; set; }
        Type this[T i0, T i1, T i2] { get; set; }
        Type this[params T[] indices] { get; set; }
    }

    /// <summary>Minimal array: no fancy stuff, no bound checking, no negative indices, no nothing</summary>
    public struct FastArray<Type> : IIndexable<int, Type>
    {
        public int Offset;

        public int[] Stride;

        public Type[] Values;

        public Type this[params int[] indices]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                var offset = Offset;
                for (int i = 0; i < indices.Length; i++) offset += indices[i] * Stride[i];
                return Values[offset];
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set
            {
                var offset = Offset;
                for (int i = 0; i < indices.Length; i++) offset += indices[i] * Stride[i];
                Values[offset] = value;
            }
        }

        public Type this[int i0]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                return Values[Offset + i0 * Stride[0]];
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set
            {
                Values[Offset + i0 * Stride[0]] = value;
            }
        }

        public Type this[int i0, int i1]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                return Values[Offset + i0 * Stride[0] + i1 * Stride[1]];
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set
            {
                Values[Offset + i0 * Stride[0] + i1 * Stride[1]] = value;
            }
        }

        public Type this[int i0, int i1, int i2]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                return Values[Offset + i0 * Stride[0] + i1 * Stride[1] + i2 * Stride[2]];
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set
            {
                Values[Offset + i0 * Stride[0] + i1 * Stride[1] + i2 * Stride[2]] = value;
            }
        }

        public Array<Type> this[Array<int> indices]
        {
            get
            {
                var thiz = this;
                var result = NN.Zeros<Type>(indices.Shape);
                Array_.ElementwiseOp(indices, result, (n, ids, offIds, incIds, res, offRes, incRes) =>
                {
                    for(int i=0; i < n; ++i)
                    {
                        res[offRes] = thiz[ids[offIds]];
                        offIds += incIds;
                        offRes += offRes;
                    }
                });
                return result;
            }
        }
    }

    public static class EmptyArray<T>
    {
        public static readonly T[] Value = new T[0];
    }

    public static class CopyExtension
    {
        public static T[] CopyToNew<T>(this T[] thiz){
            var v = new T[thiz.Length];
            thiz.CopyTo(v, 0);
            return v;
        }
    }
}
