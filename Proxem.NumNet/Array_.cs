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

namespace Proxem.NumNet
{
    using static ShapeUtil;

    public static class Array_
    {
        public static void ElementwiseOp<T1, T2>(int axisA, int axisB, int lastAxis,
            Array<T1> a, int offseta, Slice[] slices,
            Array<T2> b, int offsetb,
            Action<int, T1[], int, int, T2[], int, int> op)
        {
            var slice = axisA < slices.Length ? slices[axisA] : Slicer._;
            var lower = a.GetAbsoluteIndex(slice.Start, axisA);
            var upper = slice.IsSingleton() ? lower + 1 : a.GetAbsoluteIndex(slice.Stop, axisA);
            if (axisA == lastAxis)
            {
                int n = !slice.IsSingleton() ? (upper - lower) / slice.Step + ((upper - lower) % slice.Step == 0 ? 0 : 1) : 1;
                var strideA = a.Shape.Length == 0 ? 1 : a.Stride[axisA];
                var strideB = b.Shape.Length == 0 ? 1 : b.Stride[axisB];
                op(n, a.Values, offseta + a.Offset + lower * strideA, strideA * slice.Step, b.Values, offsetb + b.Offset, strideB);
            }
            else
            {
                offseta += lower * a.Stride[axisA];
                /*
                 * Here axisB is not incremented when slices[axisA].length == 1 (Broadcasting)
                 * But if b.Shape[axisB] == 1 there is no need to broadcast
                 * So the convention is the following:
                 *      - either A and B have the same shape,
                 *      - or B can't have any dim of shape 1.
                 */
                var incb = (slice.IsSingleton() && b.Shape[axisB] != 1) ? 0 : 1;
                if (axisB >= b.Shape.Length - 1) incb = 0;
                for (int i = lower; i < upper; i++)
                {
                    ElementwiseOp(axisA + 1, axisB + incb, lastAxis, a, offseta, slices, b, offsetb, op);
                    if (incb != 0) offseta += a.Stride[axisA];
                    offsetb += b.Stride[axisB];
                }
            }
        }

        // TODO: optimize for singletons
        public static void ElementwiseOp<T1, T2>(int axisA, int axisB, int lastAxis,
            Array<T1> a, int offseta, int[] singletons,
            Array<T2> b, int offsetb,
            Action<int, T1[], int, int, T2[], int, int> op)
        {
            var slice = axisA < singletons.Length ? singletons[axisA] : Slicer._;
            var lower = a.GetAbsoluteIndex(slice.Start, axisA);
            var upper = slice.IsSingleton() ? lower + 1 : a.GetAbsoluteIndex(slice.Stop, axisA);
            if (axisA == lastAxis)
            {
                int n = !slice.IsSingleton() ? (upper - lower) / slice.Step + ((upper - lower) % slice.Step == 0 ? 0 : 1) : 1;
                var strideA = a.Shape.Length == 0 ? 1 : a.Stride[axisA];
                var strideB = b.Shape.Length == 0 ? 1 : b.Stride[axisB];
                op(n, a.Values, offseta + a.Offset + lower * strideA, strideA * slice.Step, b.Values, offsetb + b.Offset, strideB);
            }
            else
            {
                offseta += lower * a.Stride[axisA];
                /*
                 * Here axisB is not incremented when slices[axisA].length == 1 (Broadcasting)
                 * But if b.Shape[axisB] == 1 there is no need to broadcast
                 * So the convention is the following:
                 *      - either A and B have the same shape,
                 *      - or B can't have any dim of shape 1.
                 */
                var incb = (slice.IsSingleton() && b.Shape[axisB] != 1) ? 0 : 1;
                if (axisB >= b.Shape.Length - 1) incb = 0;
                for (int i = lower; i < upper; i++)
                {
                    ElementwiseOp(axisA + 1, axisB + incb, lastAxis, a, offseta, singletons, b, offsetb, op);
                    if (incb != 0) offseta += a.Stride[axisA];
                    offsetb += b.Stride[axisB];
                }
            }
        }

        public static Array<T1> ElementwiseOp<T1>(Array<T1> a, Action<int, T1[], int, int> op)
        {
            ElementwiseOp(0, a, 0, op);
            return a;
        }

        public static void ElementwiseOp<T1>(int axis,
             Array<T1> a, int offseta,
             Action<int, T1[], int, int> op)
        {
            var lastAxis = a.Shape.Length - 1;
            if (lastAxis == -1)
            {
                op(1, a.Values, offseta + a.Offset, 1);
                return;
            }
            while (lastAxis != 0 && a.Shape[lastAxis] == 1) --lastAxis;
            if (axis == lastAxis)
            {
                op(a.Shape[axis], a.Values, offseta + a.Offset, a.Stride[axis]);
            }
            else
            {
                for (int i = 0; i < a.Shape[axis]; i++)
                {
                    ElementwiseOp(axis + 1, a, offseta, op);
                    offseta += a.Stride[axis];
                }
            }
        }

        //public static void ElementwiseOp(int axis,
        //     Array<T1> a, int offseta,
        //     Action<int, T1[], int, int> op)
        //{
        //    var lastAxis = a.Shape.Length - 1;
        //    var stride = a.Stride[lastAxis];
        //    var n = a.Shape[lastAxis];

        //    foreach(var off in a.GetOffsets(a.Shape))
        //    {
        //        op(n, a.Values, off, stride);
        //    }
        //}

        public static void ElementwiseOp<T1, T2>(Array<T1> a, Array<T2> b, Action<int, T1[], int, int, T2[], int, int> op, int opAxis = -1)
        {
            if (a.Shape.Length == 0)
            {
                op(1, a.Values, a.Offset, 1, b.Values, b.Offset, 1);
                return;
            }

            var lastAxis = a.Shape.Length - 1;
            while (lastAxis != 0 && a.Shape[lastAxis] == 1 && b.Shape[lastAxis] == 1)
                --lastAxis;
            if (opAxis < 0) opAxis += a.Shape.Length;

            ElementwiseOp(0, a, 0, b, 0, op, opAxis, lastAxis);
        }

        private static void ElementwiseOp<T1, T2>(int axis,
            Array<T1> a, int offseta,
            Array<T2> b, int offsetb,
            Action<int, T1[], int, int, T2[], int, int> op,
            int opAxis, int lastAxis)
        {
            // we have found the lastAxis, and it's the also the opAxis
            if (axis == lastAxis && lastAxis == opAxis)
                op(a.Shape[opAxis], a.Values, offseta + a.Offset, a.Stride[opAxis], b.Values, offsetb + b.Offset, b.Stride[opAxis]);
            // we found the lastAxis, so we execute the op on the opAxis
            else if (axis == lastAxis)
                for (int i = 0; i < a.Shape[axis]; i++)
                {
                    op(a.Shape[opAxis], a.Values, offseta + a.Offset, a.Stride[opAxis], b.Values, offsetb + b.Offset, b.Stride[opAxis]);
                    offseta += a.Stride[axis];
                    offsetb += b.Stride[axis];
                }
            // we skip the opAxis as the op will do the loop on the opAxis himself
            else if (axis == opAxis)
                ElementwiseOp(axis + 1, a, offseta, b, offsetb, op, opAxis, lastAxis);
            // we loop on all indexes of the current axis
            else
                for (int i = 0; i < a.Shape[axis]; i++)
                {
                    ElementwiseOp(axis + 1, a, offseta, b, offsetb, op, opAxis, lastAxis);
                    offseta += a.Stride[axis];
                    offsetb += b.Stride[axis];
                }
        }

        // requires c.Shape == BroadcastShapes(a.Shape, b.Shape)
        // if needed, enforce precondition with c = c.Reshape(BroadcastShapes(a, b))
        internal static void ElementwiseOp<T1, T2, T3>(int axis,
            Array<T1> a, int offseta,
            Array<T2> b, int offsetb,
            Array<T3> result, int offsetResult,
            Action<int, T1[], int, int, T2[], int, int, T3[], int, int> op)
        {
            Debug.Assert(CheckShapes(a, b, result));

            // It holds that
            Debug.Assert(result.Shape.Length >= a.Shape.Length);
            Debug.Assert(result.Shape.Length >= b.Shape.Length);

            var lastAxisA = a.Shape.Length - 1;
            var lastAxisB = b.Shape.Length - 1;
            var lastAxisC = result.Shape.Length - 1;
            while (lastAxisC != 0 && result.Shape[lastAxisC] == 1)
            {
                // Invariant: ForAll (axisA, axisC) : (c.Shape[axisC] == 1) => (a.Shape[axisA] == 1)
                Debug.Assert(lastAxisA < 0 || a.Shape[lastAxisA] == 1);     // from invariant
                Debug.Assert(lastAxisB < 0 || b.Shape[lastAxisB] == 1);
                --lastAxisA;
                --lastAxisB;
                --lastAxisC;
            }
            var axisA = lastAxisA - (lastAxisC - axis);     // axes are aligned from right to left
            var axisB = lastAxisB - (lastAxisC - axis);
            if (axis == lastAxisC)
            {
                // if (a.Shape[axisA] == 1 && c.Shape[axis] != 1) => broadcasting => strideA == 0
                // NB: c.Shape[axis] == 1 is possible if axis == 0
                var strideA = (axisA < 0 || a.Shape[axisA] == 1) && result.Shape[axis] != 1 ? 0 : a.Stride[axisA];
                var strideB = (axisB < 0 || b.Shape[axisB] == 1) && result.Shape[axis] != 1 ? 0 : b.Stride[axisB];
                op(result.Shape[axis], a.Values, offseta + a.Offset, strideA, b.Values, offsetb + b.Offset, strideB, result.Values, offsetResult + result.Offset, result.Stride[axis]);
            }
            else
            {
                // broadcasting
                var dimA = axisA < 0 ? 1 : a.Shape[axisA];
                var dimB = axisB < 0 ? 1 : b.Shape[axisB];
                var dimC = result.Shape[axis];
                Debug.Assert(dimA == 1 || dimA == dimC);    // from invariant
                Debug.Assert(dimB == 1 || dimB == dimC);    // from invariant

                for (int i = 0; i < dimC; i++)
                {
                    ElementwiseOp(axis + 1, a, offseta, b, offsetb, result, offsetResult, op);
                    if (dimA != 1) offseta += a.Stride[axis];       // if a is broadcast, don't move
                    if (dimB != 1) offsetb += b.Stride[axis];       // if b is broadcast, don't move
                    offsetResult += result.Stride[axis];
                }
            }
        }

        internal static void ElementwiseOp<T1, T2, T3, R>(int axis,
            Array<T1> a, int offseta,
            Array<T2> b, int offsetb,
            Array<T3> c, int offsetc,
            Array<R> result, int offsetResult,
            Action<int, T1[], int, int, T2[], int, int, T3[], int, int, R[], int, int> op)
        {
            Debug.Assert(CheckShapes(a, b, result));

            // It holds that
            Debug.Assert(result.Shape.Length >= a.Shape.Length);
            Debug.Assert(result.Shape.Length >= b.Shape.Length);
            Debug.Assert(result.Shape.Length >= c.Shape.Length);

            var lastAxisA = a.Shape.Length - 1;
            var lastAxisB = b.Shape.Length - 1;
            var lastAxisC = c.Shape.Length - 1;
            var lastAxisResult = result.Shape.Length - 1;
            while (lastAxisResult != 0 && result.Shape[lastAxisResult] == 1)
            {
                // Invariant: ForAll (axisA, axisC) : (c.Shape[axisC] == 1) => (a.Shape[axisA] == 1)
                Debug.Assert(lastAxisA < 0 || a.Shape[lastAxisA] == 1);     // from invariant
                Debug.Assert(lastAxisB < 0 || b.Shape[lastAxisB] == 1);
                Debug.Assert(lastAxisC < 0 || c.Shape[lastAxisC] == 1);
                --lastAxisA;
                --lastAxisB;
                --lastAxisC;
                --lastAxisResult;
            }
            var axisA = lastAxisA - (lastAxisResult - axis);     // axes are aligned from right to left
            var axisB = lastAxisB - (lastAxisResult - axis);
            var axisC = lastAxisC - (lastAxisResult - axis);
            if (axis == lastAxisResult)
            {
                // if (a.Shape[axisA] == 1 && c.Shape[axis] != 1) => broadcasting => strideA == 0
                // NB: c.Shape[axis] == 1 is possible if axis == 0
                var strideA = (axisA < 0 || a.Shape[axisA] == 1) && result.Shape[axis] != 1 ? 0 : a.Stride[axisA];
                var strideB = (axisB < 0 || b.Shape[axisB] == 1) && result.Shape[axis] != 1 ? 0 : b.Stride[axisB];
                var strideC = (axisC < 0 || c.Shape[axisC] == 1) && result.Shape[axis] != 1 ? 0 : c.Stride[axisC];
                op(result.Shape[axis], a.Values, offseta + a.Offset, strideA, b.Values, offsetb + b.Offset, strideB, c.Values, offsetc + c.Offset, strideC, result.Values, offsetResult + result.Offset, result.Stride[axis]);
            }
            else
            {
                // broadcasting
                var dimA = axisA < 0 ? 1 : a.Shape[axisA];
                var dimB = axisB < 0 ? 1 : b.Shape[axisB];
                var dimC = axisC < 0 ? 1 : c.Shape[axisC];
                var dimR = result.Shape[axis];
                Debug.Assert(dimA == 1 || dimA == dimR);    // from invariant
                Debug.Assert(dimB == 1 || dimB == dimR);    // from invariant
                Debug.Assert(dimC == 1 || dimC == dimR);    // from invariant

                for (int i = 0; i < dimR; i++)
                {
                    ElementwiseOp(axis + 1, a, offseta, b, offsetb, c, offsetc, result, offsetResult, op);
                    if (dimA != 1) offseta += a.Stride[axis];       // if a is broadcast, don't move
                    if (dimB != 1) offsetb += b.Stride[axis];       // if b is broadcast, don't move
                    if (dimC != 1) offsetc += c.Stride[axis];       // if c is broadcast, don't move
                    offsetResult += result.Stride[axis];
                }
            }
        }

        internal static void ElementwiseOp<T1, T2, T3, T4, R>(int axis,
            Array<T1> a, int offseta,
            Array<T2> b, int offsetb,
            Array<T3> c, int offsetc,
            Array<T4> d, int offsetd,
            Array<R> result, int offsetResult,
            Action<int, T1[], int, int, T2[], int, int, T3[], int, int, T4[], int, int, R[], int, int> op)
        {
            Debug.Assert(CheckShapes(a, b, result));

            // It holds that
            Debug.Assert(result.Shape.Length >= a.Shape.Length);
            Debug.Assert(result.Shape.Length >= b.Shape.Length);
            Debug.Assert(result.Shape.Length >= c.Shape.Length);
            Debug.Assert(result.Shape.Length >= d.Shape.Length);

            var lastAxisA = a.Shape.Length - 1;
            var lastAxisB = b.Shape.Length - 1;
            var lastAxisC = c.Shape.Length - 1;
            var lastAxisD = d.Shape.Length - 1;
            var lastAxisResult = result.Shape.Length - 1;
            while (lastAxisResult != 0 && result.Shape[lastAxisResult] == 1)
            {
                // Invariant: ForAll (axisA, axisC) : (c.Shape[axisC] == 1) => (a.Shape[axisA] == 1)
                Debug.Assert(lastAxisA < 0 || a.Shape[lastAxisA] == 1);     // from invariant
                Debug.Assert(lastAxisB < 0 || b.Shape[lastAxisB] == 1);
                Debug.Assert(lastAxisC < 0 || c.Shape[lastAxisC] == 1);
                Debug.Assert(lastAxisD < 0 || d.Shape[lastAxisD] == 1);
                --lastAxisA;
                --lastAxisB;
                --lastAxisC;
                --lastAxisD;
                --lastAxisResult;
            }
            var axisA = lastAxisA - (lastAxisResult - axis);     // axes are aligned from right to left
            var axisB = lastAxisB - (lastAxisResult - axis);
            var axisC = lastAxisC - (lastAxisResult - axis);
            var axisD = lastAxisD - (lastAxisResult - axis);
            if (axis == lastAxisResult)
            {
                // if (a.Shape[axisA] == 1 && c.Shape[axis] != 1) => broadcasting => strideA == 0
                // NB: c.Shape[axis] == 1 is possible if axis == 0
                var strideA = (axisA < 0 || a.Shape[axisA] == 1) && result.Shape[axis] != 1 ? 0 : a.Stride[axisA];
                var strideB = (axisB < 0 || b.Shape[axisB] == 1) && result.Shape[axis] != 1 ? 0 : b.Stride[axisB];
                var strideC = (axisC < 0 || c.Shape[axisC] == 1) && result.Shape[axis] != 1 ? 0 : c.Stride[axisC];
                var strideD = (axisD < 0 || d.Shape[axisD] == 1) && result.Shape[axis] != 1 ? 0 : d.Stride[axisD];
                op(result.Shape[axis],
                    a.Values, offseta + a.Offset, strideA,
                    b.Values, offsetb + b.Offset, strideB,
                    c.Values, offsetc + c.Offset, strideC,
                    d.Values, offsetd + d.Offset, strideD,
                    result.Values, offsetResult + result.Offset, result.Stride[axis]
                );
            }
            else
            {
                // broadcasting
                var dimA = axisA < 0 ? 1 : a.Shape[axisA];
                var dimB = axisB < 0 ? 1 : b.Shape[axisB];
                var dimC = axisC < 0 ? 1 : c.Shape[axisC];
                var dimD = axisD < 0 ? 1 : d.Shape[axisD];
                var dimR = result.Shape[axis];
                Debug.Assert(dimA == 1 || dimA == dimR);    // from invariant
                Debug.Assert(dimB == 1 || dimB == dimR);    // from invariant
                Debug.Assert(dimC == 1 || dimC == dimR);    // from invariant
                Debug.Assert(dimD == 1 || dimD == dimR);    // from invariant

                for (int i = 0; i < dimR; i++)
                {
                    ElementwiseOp(axis + 1, a, offseta, b, offsetb, c, offsetc, d, offsetd, result, offsetResult, op);
                    if (dimA != 1) offseta += a.Stride[axis];       // if a is broadcast, don't move
                    if (dimB != 1) offsetb += b.Stride[axis];       // if b is broadcast, don't move
                    if (dimC != 1) offsetc += c.Stride[axis];       // if c is broadcast, don't move
                    if (dimD != 1) offsetd += d.Stride[axis];       // if d is broadcast, don't move
                    offsetResult += result.Stride[axis];
                }
            }
        }

        internal static Array<T3> ElementwiseOp<T1, T2, T3>(Array<T1> a, Array<T2> b, Array<T3> result,
            Action<int, T1[], int, int, T2[], int, int, T3[], int, int> op)
        {
            if (result == null)
            {
                result = new Array<T3>(BroadcastShapes(a, b));
            }
            else if (!CheckShapes(a, b, result))
            {
                throw new RankException(string.Format("Incompatible result shape, expected ({0}) got ({1})",
                    string.Join(", ", BroadcastShapes(a, b)), string.Join(", ", result.Shape)));
            }

            // if a, b and result are contiguous AND there is no broadcasting (all sizes are equal), we can use op directly
            if (a.IsContiguous() && b.IsContiguous() && result.IsContiguous() && result.Size == a.Size && result.Size == b.Size)
            {
                var size = result.Size;
                if (size == a.Size && size == b.Size)
                {
                    if (size == 1)
                        op(size, a.Values, a.Offset, 1, b.Values, b.Offset, 1, result.Values, result.Offset, 1);
                    else
                        op(size, a.Values, a.Offset, a.LastStride(), b.Values, b.Offset, b.LastStride(), result.Values, result.Offset, result.LastStride());
                    return result;
                }
            }
            // else we have to iterate on the dim of the arrays
            // NB: ElementwiseOp's precondition is enforced above
            ElementwiseOp(0, a, 0, b, 0, result, 0, op);
            return result;
        }

        internal static Array<R> ElementwiseOp<T1, T2, T3, R>(Array<T1> a, Array<T2> b, Array<T3> c, Array<R> result,
            Action<int, T1[], int, int, T2[], int, int, T3[], int, int, R[], int, int> op)
        {
            if (result == null)
            {
                result = new Array<R>(BroadcastShapes(BroadcastShapes(a, b), c));
            }
            else if (!CheckShapes(a, b, result))
            {
                throw new RankException(string.Format("Incompatible result shape, expected ({0}) got ({1})",
                    string.Join(", ", BroadcastShapes(a, b)), string.Join(", ", result.Shape)));
            }

            // if a, b and result are contiguous AND there is no broadcasting (all sizes are equal), we can use op directly
            if (a.IsContiguous() && b.IsContiguous() && c.IsContiguous() && result.IsContiguous() && result.Size == a.Size && result.Size == b.Size && result.Size == c.Size)
            {
                var size = result.Size;
                if (size == a.Size && size == b.Size)
                {
                    if (a.Shape.Length == 0)
                        op(size, a.Values, a.Offset, 1, b.Values, b.Offset, 1, c.Values, c.Offset, 1, result.Values, result.Offset, 1);
                    else
                        op(size, a.Values, a.Offset, a.LastStride(), b.Values, b.Offset, b.LastStride(), c.Values, c.Offset, c.LastStride(), result.Values, result.Offset, result.LastStride());
                    return result;
                }
            }
            // else we have to iterate on the dim of the arrays
            // NB: ElementwiseOp's precondition is enforced above
            ElementwiseOp(0, a, 0, b, 0, c, 0, result, 0, op);
            return result;
        }

        internal static Array<R> ElementwiseOp<T1, T2, T3, T4, R>(Array<T1> a, Array<T2> b, Array<T3> c, Array<T4> d, Array<R> result,
            Action<int, T1[], int, int, T2[], int, int, T3[], int, int, T4[], int, int, R[], int, int> op)
        {
            var resultShape = BroadcastShapes(BroadcastShapes(BroadcastShapes(a, b), c), d);
            if (result == null)
                result = new Array<R>(resultShape);
            else
                result.AssertOfShape(resultShape);

            // if a, b and result are contiguous AND there is no broadcasting (all sizes are equal), we can use op directly
            if (a.IsContiguous() && b.IsContiguous() && c.IsContiguous() && d.IsContiguous() && result.IsContiguous()
                && result.Size == a.Size && result.Size == b.Size && result.Size == c.Size && result.Size == d.Size)
            {
                var size = result.Size;
                if (size == a.Size && size == b.Size)
                {
                    if (a.Shape.Length == 0)
                        op(size, a.Values, a.Offset, 1, b.Values, b.Offset, 1, c.Values, c.Offset, 1, d.Values, d.Offset, 1, result.Values, result.Offset, 1);
                    else
                        op(size,
                            a.Values, a.Offset, a.LastStride(),
                            b.Values, b.Offset, b.LastStride(),
                            c.Values, c.Offset, c.LastStride(),
                            d.Values, d.Offset, d.LastStride(),
                            result.Values, result.Offset, result.LastStride());
                    return result;
                }
            }
            // else we have to iterate on the dim of the arrays
            // NB: ElementwiseOp's precondition is enforced above
            ElementwiseOp(0, a, 0, b, 0, c, 0, d, 0, result, 0, op);
            return result;
        }

        public static IEnumerable<Array<T>> Rows<T>(this Array<T> array, int axis = 0, bool keepDims = false)
        {
            if (axis < 0) axis += array.NDim;
            var slices = array.Slices();
            slices[axis] = keepDims ? Slicer.Range(0, 1) : 0;

            int N = array.Shape[axis];
            int stride = array.Stride[axis];
            var offset = array.Offset;

            var first = array[slices];
            yield return first;

            for (int i = 1; i < N; ++i)
            {
                offset += stride;
                yield return new Array<T>(first.Shape, first.Values, offset, first.Stride);
            }
        }

        public static IEnumerable<Array<T>> UnsafeRows<T>(this Array<T> array, int axis = 0, bool keepDims = false)
        {
            if (axis < 0) axis += array.NDim;
            var slices = array.Slices();
            slices[axis] = keepDims ? Slicer.Range(0, 1) : 0;

            int N = array.Shape[axis];
            int stride = array.Stride[axis];
            var _current = array[slices];

            for (int i = 0; i < N; ++i)
            {
                yield return _current;
                _current.Offset += stride;
            }
        }
    }
}
