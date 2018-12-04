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

namespace Proxem.NumNet
{
    public static class ShapeUtil
    {
        public static int[] RemoveAxis(int[] shape, int axis)
        {
            var resultShape = new int[shape.Length - 1];
            Array.Copy(shape, 0, resultShape, 0, axis);
            if (axis < shape.Length - 1)
                Array.Copy(shape, axis + 1, resultShape, axis, shape.Length - axis - 1);
            return shape;
        }

        public static Array<T> Reshape<T>(this Array<T> a, params int[] shape)
        {
            return a.Reshape(shape);
        }

        public static Array<T> Transpose<T>(this Array<T> a, params int[] axes)
        {
            return a.Transpose(axes);
        }

        internal static int[] BroadcastShapes<T2>(int[] a, Array<T2> b)
        {
            return BroadcastShapes(a, b.Shape);
        }

        internal static int[] BroadcastShapes<T1, T2>(Array<T1> a, Array<T2> b)
        {
            return BroadcastShapes(a.Shape, b.Shape);
        }

        // broadcasting: http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
        // When operating on two arrays, NumPy compares their shapes element-wise.
        // It starts with the trailing dimensions, and works its way forward. Two dimensions are compatible when
        // 1. they are equal, or
        // 2. one of them is 1
        // If these conditions are not met, a ValueError: frames are not aligned exception is thrown, indicating that the arrays have
        // incompatible shapes. The size of the resulting array is the maximum size along each dimension of the input arrays.
        public static int[] BroadcastShapes(int[] a, int[] b)
        {
            var lengthA = a.Length;
            var lengthB = b.Length;
            if (lengthA < lengthB) return BroadcastShapes(b, a);

            // new shape's size is largest shape's size
            var newShape = new int[lengthA];

            // align a and b on the right
            // if a has trailing 1s, broadcast b on the right
            //      a = [x y z 1 1]
            //      b =   [m n o p]
            // result = [. . . o p]
            while (lengthB > 0 && lengthA > lengthB && a[lengthA - 1] == 1)
            {
                --lengthA;
                --lengthB;
                newShape[lengthA] = b[lengthB];
            }

            // broadcast a on the left
            //      a = [x y z | 1 1]
            //      b =   [m n | o p]
            // result = [x . . | o p]
            for (int i = 0; i < lengthA - lengthB; i++) newShape[i] = a[i];

            // for aligned axes, either y == m (resp. z == n) or one of them is 1
            for (int i = lengthA - lengthB; i < lengthA; i++)
            {
                var shapeA = a[i];
                var shapeB = b[i - (lengthA - lengthB)];
                if (shapeA != 1 && shapeB != 1 && shapeA != shapeB)
                    throw new RankException(string.Format("operands could not be broadcast together with shapes ({0}) and ({1})",
                        string.Join(", ", a), string.Join(", ", b)));
                newShape[i] = shapeA != 1 ? shapeA : shapeB;
            }
            return newShape;
        }

        internal static bool CheckShapes(int[] a, int[] b)
        {
            var lengthA = a.Length;
            var lengthB = b.Length;
            if (lengthA < lengthB) return CheckShapes(b, a);

            // align a and b on the right
            // if a has trailing 1s, broadcast b on the right
            //      a = [x y z 1 1]
            //      b =   [m n o p]
            while (lengthB > 0 && lengthA > lengthB && a[lengthA - 1] == 1)
            {
                --lengthA;
                --lengthB;
            }

            // for aligned axes, either y == m (resp. z == n) or one of them is 1
            for (int i = lengthA - lengthB; i < lengthA; i++)
            {
                var shapeA = a[i];
                var shapeB = b[i - (lengthA - lengthB)];
                if (shapeA != 1 && shapeB != 1 && shapeA != shapeB) return false;
            }
            return true;
        }


        // check that result.Shape == BroadcastShape(a, b) without creating an intermediate array
        internal static bool CheckShapes<T1, T2, T3>(Array<T1> a, Array<T2> b, Array<T3> result)
        {
            return CheckShapes(a.Shape, b.Shape, result.Shape);
        }

        internal static bool CheckShapes(int[] a, int[] b, int[] result)
        {
            var lengthA = a.Length;
            var lengthB = b.Length;
            if (lengthA < lengthB) return CheckShapes(b, a, result);

            // result shape's size is largest shape's size
            if (result.Length != a.Length) return false;

            // align a and b on the right
            // if a has trailing 1s, broadcast b on the right
            //      a = [x y z 1 1]
            //      b =   [m n o p]
            // result = [. . . o p]
            while (lengthB > 0 && lengthA > lengthB && a[lengthA - 1] == 1)
            {
                --lengthA;
                --lengthB;
                if (result[lengthA] != b[lengthB]) return false;
            }

            // broadcast a on the left
            //      a = [x y z | 1 1]
            //      b =   [m n | o p]
            // result = [x . . | o p]
            for (int i = 0; i < lengthA - lengthB; i++)
            {
                if (result[i] != a[i]) return false;
            }

            // for aligned axes, either y == m (resp. z == n) or one of them is 1
            for (int i = lengthA - lengthB; i < lengthA; i++)
            {
                var shapeA = a[i];
                var shapeB = b[i - (lengthA - lengthB)];
                if (shapeA != 1 && shapeB != 1 && shapeA != shapeB)
                    throw new RankException(string.Format("operands could not be broadcast together with shapes ({0}) and ({1})",
                        string.Join(", ", a), string.Join(", ", b)));
                if (result[i] != (shapeA != 1 ? shapeA : shapeB)) return false;
            }
            return true;
        }

        public static int[] GetAggregatorResultShape<T>(Array<T> a, int axis, bool keepDims)
        {
            var extra = keepDims ? 1 : 0;
            int[] shape = new int[a.Shape.Length - 1 + extra];
            Array.Copy(a.Shape, 0, shape, 0, axis);
            if (keepDims) shape[axis] = 1;
            Array.Copy(a.Shape, axis + 1, shape, axis + extra, shape.Length - axis - extra);
            return shape;
        }
    }
}
