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

namespace Proxem.NumNet
{
    using System.Runtime.CompilerServices;
    using static StridedExtension;

    public class Strided<Type>
    {
        /// <summary>Shape of the arrays, ie the size of each axis of the array.</summary>
        public int[] Shape;

        /// <summary>The index of the first value of this Array in the underlying data</summary>
        public int Offset;

        /// <summary>
        /// For each axis the number of step needed in the underlying data array to increase this axis coordinate of 1.
        /// Contrary to NumPy the stride isn't measured in byte.
        /// See <a href="http://en.wikipedia.org/wiki/Stride_of_an_array">wikipedia</a> for mor details on stride.
        /// </summary>
        public int[] Stride;

        /// <summary>Information about the underlying data storage</summary>
        public Flags Flags;

        /// <summary>The number of elements in this Array</summary>
        public int Size => ComputeSize(this.Shape);

        /// <summary>The number of dimensions in this Array</summary>
        public int NDim { [MethodImpl(MethodImplOptions.AggressiveInlining)] get { return Shape.Length; } }

        public int GetAbsoluteIndex(int index, int axis)
        {
            if (index == int.MaxValue)
                return this.Shape[axis];
            else if (index == int.MinValue)
                return -1;
            else
                return index >= 0 ? index : this.Shape[axis] + index;
        }

        public int GetAbsoluteIndex(Index index, int axis)
        {
            //return GetAbsoluteIndex(index.IsFromEnd ? -index.Value : index.Value, axis);
            if (index.IsFromEnd)
                return this.Shape[axis] - index.Value;
            else if (index.Value == Slice.MinusOne)
                return -1;
            else
                return index.Value;
        }

        /// <summary> Convert coordinates to offset </summary>
        public int RavelIndices(int i0)
        {
            if (1 > this.Shape.Length) throw new Exception("too many indices");
            var dim0 = GetAbsoluteIndex(i0, 0);
            if (dim0 >= this.Shape[0]) throw new IndexOutOfRangeException($"index {i0} out of range 0<=index<{Shape[0]}");
            return this.Offset + dim0 * this.Stride[0];
        }

        public int RavelIndices(int i0, int i1)
        {
            if (2 > this.Shape.Length) throw new Exception("too many indices");
            var dim0 = GetAbsoluteIndex(i0, 0);
            if (dim0 >= this.Shape[0]) throw new IndexOutOfRangeException($"index {i0} out of range 0<=index<{Shape[0]}");
            var dim1 = GetAbsoluteIndex(i1, 1);
            if (dim1 >= this.Shape[1]) throw new IndexOutOfRangeException($"index {i1} out of range 0<=index<{Shape[1]}");
            return this.Offset + dim0 * this.Stride[0] + dim1 * this.Stride[1];
        }

        public int RavelIndices(int i0, int i1, int i2)
        {
            if (3 > this.Shape.Length) throw new Exception("too many indices");
            var dim0 = GetAbsoluteIndex(i0, 0);
            if (dim0 >= this.Shape[0]) throw new IndexOutOfRangeException($"index {i0} out of range 0<=index<{Shape[0]}");
            var dim1 = GetAbsoluteIndex(i1, 1);
            if (dim1 >= this.Shape[1]) throw new IndexOutOfRangeException($"index {i1} out of range 0<=index<{Shape[1]}");
            var dim2 = GetAbsoluteIndex(i2, 2);
            if (dim2 >= this.Shape[2]) throw new IndexOutOfRangeException($"index {i2} out of range 0<=index<{Shape[2]}");
            return this.Offset + dim0 * this.Stride[0] + dim1 * this.Stride[1] + dim2 * this.Stride[2];
        }

        public int RavelIndices(params int[] indices)
        {
            if (indices.Length > this.Shape.Length) throw new Exception("too many indices");
            int result = this.Offset;
            for (int axis = 0; axis < indices.Length; axis++)
            {
                var dim = GetAbsoluteIndex(indices[axis], axis);
                if (dim >= this.Shape[axis]) throw new IndexOutOfRangeException($"index {indices} out of range 0<=index<{Shape[axis]}");
                result += dim * this.Stride[axis];
            }
            return result;
        }

        public int RavelIndices(params Index[] indices)
        {
            if (indices.Length > this.Shape.Length) throw new Exception("too many indices");
            int result = this.Offset;
            for (int axis = 0; axis < indices.Length; axis++)
            {
                var dim = GetAbsoluteIndex(indices[axis], axis);
                if (dim >= this.Shape[axis]) throw new IndexOutOfRangeException($"index {indices} out of range 0<=index<{Shape[axis]}");
                result += dim * this.Stride[axis];
            }
            return result;
        }

        protected int RavelIndicesStart(Slice[] slices)
        {
            if (slices.Length > this.Shape.Length) throw new Exception("too many indices");
            int result = this.Offset;
            for (int axis = 0; axis < slices.Length; axis++)
            {
                var dim = GetAbsoluteIndex(slices[axis].Range.Start, axis);
                if (dim >= this.Shape[axis]) throw new IndexOutOfRangeException($"index {slices} out of range 0<=index<{Shape[axis]}");
                result += dim * this.Stride[axis];
            }
            return result;
        }

        protected int RavelIndicesStart(Slice?[] slices)
        {
            if (slices.Length > this.Shape.Length) throw new Exception("too many indices");
            int result = this.Offset;
            for (int axis = 0; axis < slices.Length; axis++)         
            {
                ref var slice = ref slices[axis];
                if (slice == null) continue;
                var dim = GetAbsoluteIndex(slice.Value.Range.Start, axis);
                if (dim >= this.Shape[axis]) throw new IndexOutOfRangeException($"index {slices} out of range 0<=index<{Shape[axis]}");
                result += dim * this.Stride[axis];
            }
            return result;
        }

        /// <summary>
        /// Yields the coordinate of the i-th value traversed by this tensor enumerator.
        /// Examples:
        /// <c>
        ///   var t = Tensor.Zeros(20, 15);
        ///   var coord = t.Unravel(43);  // = new int[]{2, 13};
        /// </c>
        /// </summary>
        public int[] UnravelIndex(int i, int[] result = null)
        {
            if (result == null) result = new int[Shape.Length];
            for (int axis = Shape.Length - 1; axis >= 0; --axis)
            {
                result[axis] = i % Shape[axis];
                i = i / Shape[axis];
            }
            return result;
        }
    }

    public static class StridedExtension
    {
        public static Slice[] Slices<T>(this Strided<T> a) => a.Shape.Select(d => (Slice)(0..d)).ToArray();

        public static int ComputeSize(int[] shape)
        {
            var result = 1;
            for (int i = 0; i < shape.Length; i++)
            {
                result *= shape[i];
            }
            return result;
        }

        public static int[] ComputeStride(int[] shape, int[] result = null)
        {
            result = result ?? new int[shape.Length];
            var last = result.Length - 1;
            if (last == -1) return result;
            result[last] = 1;
            for (int i = last - 1; i >= 0; i--)
            {
                result[i] = result[i + 1] * shape[i + 1];
            }
            return result;
        }
    }
}
