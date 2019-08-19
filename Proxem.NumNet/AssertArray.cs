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
    public static class AssertArray
    {
        public static Exception BadRank(string message, params object[] content)
        {
            return new RankException(string.Format(message, content));
        }

        public static Exception Failed(string message, params object[] content)
        {
            return new ArgumentException(string.Format(message, content));
        }

        public static string FormatShape(params int[] shape)
        {
            return string.Format("[{0}]", string.Join(", ", shape));
        }

        public static void AreEqual(int expected, int actual)
        {
            if (expected != actual)
                throw Failed("AssertArray.AreEqual failed. The values don't match. Expected:<{0}>. Actual:<{1}>",
                    expected, actual);
        }

        public static void IsGreaterThan(double value, double lowerBound)
        {
            if (!(value > lowerBound))
                throw Failed("AssertArray.IsGreaterThan failed. <{0}> is not greater than <{1}>", value, lowerBound);
        }

        public static void IsLessThan(double value, double upperBound)
        {
            if (!(value < upperBound))
                throw Failed("AssertArray.IsLessThan failed. <{0}> is not less than <{1}>", value, upperBound);
        }

        public static void AssertIsContiguous<T>(this Array<T> arr)
        {
            if (!arr.IsContiguous())
                throw Failed("AssertArray.AssertIsContiguous failed. This array isn't contiguous");
        }

        public static void AssertIsNotContiguous<T>(this Array<T> arr)
        {
            if (arr.IsContiguous())
                throw Failed("AssertArray.AssertIsNotContiguous failed. This array is contiguous");
        }

        public static void AssertIsTransposed<T>(this Array<T> arr)
        {
            if (!arr.IsTransposed())
                throw Failed("AssertArray.AssertIsTransposed failed. This array isn't transposed");
        }

        public static void AssertIsNotTransposed<T>(this Array<T> arr)
        {
            if (arr.IsTransposed())
                throw Failed("AssertArray.AssertIsNotTransposed failed. This array is transposed");
        }

        public static void AreEqual<T>(T[] expected, T[] actual)
        {
            if (expected.Length != actual.Length)
                throw BadRank("AssertArray.AreEqual failed. Dims don't match. Expected:<{0}>. Actual:<{1}>",
                    expected.Length, actual.Length);

            for (int i = 0; i < expected.Length; ++i)
            {
                if (!expected[i].Equals(actual[i]))
                    throw Failed("AssertArray.AreEqual failed. The {0}-th values don't match Expected:<{1}>. Actual:<{2}>",
                        i, expected[i], actual[i]);
            }
        }

        public static void AreEqual<T>(IEnumerable<T> expected, IEnumerable<T> actual)
        {
            AreEqual(expected.ToArray(), actual.ToArray());
        }

        public static void WriteTheSame<T1, T2>(IEnumerable<T1> expected, IEnumerable<T2> actual)
        {
            AreEqual(expected.Select(x => x.ToString()), actual.Select(x => x.ToString()));
        }

        public static void AreEqual<T>(Array<T> expected, Array<T> actual, bool allowBroadcasting = false)
        {
            if (allowBroadcasting)
            {
                expected = expected.MinifyDim();
                actual = actual.MinifyDim();
                if (!ShapeUtil.CheckShapes(actual.Shape, expected.Shape))
                    throw BadRank("Expected shape: {0}, actual: {1}.",
                        FormatShape(expected.Shape), FormatShape(actual.Shape));
            }
            else
            {
                actual.AssertOfShape(expected.Shape);
            }

            var eq = EqualityComparer<T>.Default;
            foreach (var xyi in expected.Zip(actual, Tuple.Create).Select(Tuple.Create<Tuple<T, T>, int>))
            {
                int i = xyi.Item2;
                var x = xyi.Item1.Item1;
                var y = xyi.Item1.Item2;
                if (!eq.Equals(x, y))
                {
                    var coord = expected.UnravelIndex(i);
                    if (expected.Size < 15)
                        throw Failed("AssertArray.AreEqual failed. Expected:<{0}>. Actual:<{1}>", expected, actual);
                    else
                        throw Failed("AssertArray.AreEqual failed. The values at [{0}] don't match. Expected:<{1}>. Actual:<{2}>",
                            coord.Aggregate("", (s, j) => s + ", " + j).Substring(2), x, y);
                }
            }
        }

        public static void AreNotEqual<T>(Array<T> expected, Array<T> actual, bool allowBroadcasting = false)
        {
            try
            {
                AreEqual(expected, actual, allowBroadcasting);
            }
            catch (RankException) { return; }
            catch (ArgumentException)   // TODO: remove try/catch
            {
                return;
            }
            throw Failed("AssertArray.AreNotEqual failed. Actual:<{0}>", actual);
        }

        public static void AreAlmostEqual(Array<float> expected, Array<float> actual, float relativeErr = 1e-6f, float absErr = 1e-6f, bool allowBroadcasting = false)
        {
            if (allowBroadcasting)
            {
                expected = expected.MinifyDim();
                actual = actual.MinifyDim();
            }

            actual.AssertOfShape(expected.Shape);

            foreach (var xyi in expected.Zip(actual, Tuple.Create).Select(Tuple.Create<Tuple<float, float>, int>))
            {
                int i = xyi.Item2;
                var x = xyi.Item1.Item1;
                var y = xyi.Item1.Item2;
                if (!CheckAreAlmostEqual(x, y, relativeErr, absErr))
                {
                    var coord = expected.UnravelIndex(i);
                    if (expected.Size <= 15)
                        throw Failed("AssertArray.AreEqual failed. Expected:<{0}>. Actual:<{1}>", expected, actual);
                    else
                        throw Failed("AssertArray.AreEqual failed. The values at {0} don't match. Expected:<{1}>. Actual:<{2}>",
                            FormatShape(coord), x, y);
                }
            }
        }

        public static void AreAlmostEqual(Array<double> expected, Array<double> actual, double relativeErr = 1e-6f, double absErr = 1e-6f, bool allowBroadcasting = false)
        {
            if (allowBroadcasting)
            {
                expected = expected.MinifyDim();
                actual = actual.MinifyDim();
            }

            actual.AssertOfShape(expected.Shape);

            foreach (var xyi in expected.Zip(actual, Tuple.Create).Select(Tuple.Create<Tuple<double, double>, int>))
            {
                int i = xyi.Item2;
                var x = xyi.Item1.Item1;
                var y = xyi.Item1.Item2;
                if (!CheckAreAlmostEqual(x, y, relativeErr, absErr))
                {
                    var coord = expected.UnravelIndex(i);
                    if (expected.Size <= 15)
                        throw Failed("AssertArray.AreEqual failed. Expected:<{0}>. Actual:<{1}>", expected, actual);
                    else
                        throw Failed("AssertArray.AreEqual failed. The values at {0} don't match. Expected:<{1}>. Actual:<{2}>",
                            FormatShape(coord), x, y);
                }
            }
        }

        static public bool CheckAreAlmostEqual(float expected, float actual, float relativeErr = 1e-6f, float absErr = 1e-6f)
        {
            if (expected == actual) return true;

            var diff = Math.Abs(expected - actual);
            var relativeDiff = 2 * diff / (Math.Abs(expected) + Math.Abs(actual));

            return diff < absErr || relativeDiff < relativeErr;
        }

        static public bool CheckAreAlmostEqual(double expected, double actual, double relativeErr = 1e-6f, double absErr = 1e-6f)
        {
            if (expected == actual) return true;

            var diff = Math.Abs(expected - actual);
            var relativeDiff = 2 * diff / (Math.Abs(expected) + Math.Abs(actual));

            return diff < absErr || relativeDiff < relativeErr;
        }

        public static void AreAlmostEqual(float expected, float actual, float relativeErr = 1e-6f, float absErr = 1e-6f)
        {
            if (!CheckAreAlmostEqual(expected, actual, relativeErr, absErr))
                throw Failed("Expected: {0}, actual {1}, diff {2}, relative {3}.",
                    expected, actual, Math.Abs(expected - actual), 2 * Math.Abs(expected - actual) / (Math.Abs(expected) + Math.Abs(actual)));
        }

        public static void AssertOfShape<T>(this Array<T> arr, params int[] shape)
        {
            if (arr.NDim != shape.Length || Enumerable.Range(0, arr.NDim).Any(i => arr.Shape[i] != shape[i]))
                throw BadRank("Expected shape: {0}, actual: {1}.",
                    FormatShape(shape), FormatShape(arr.Shape));
        }

        // inline expansion of params int[1]
        public static void AssertOfShape<T>(this Array<T> arr, int dim)
        {
            if (arr.NDim != 1 || arr.Shape[0] != dim)
                throw BadRank("Expected shape: {0}, actual: {1}.",
                    FormatShape(dim), FormatShape(arr.Shape));
        }

        // inline expansion of params int[2]
        public static void AssertOfShape<T>(this Array<T> arr, int dim1, int dim2)
        {
            if (arr.NDim != 2 || arr.Shape[0] != dim1 || arr.Shape[1] != dim2)
                throw BadRank("Expected shape: {0}, actual: {1}.",
                    FormatShape(dim1, dim2), FormatShape(arr.Shape));
        }

        // inline expansion of params int[3]
        public static void AssertOfShape<T>(this Array<T> arr, int dim1, int dim2, int dim3)
        {
            if (arr.NDim != 3 || arr.Shape[0] != dim1 || arr.Shape[1] != dim2 || arr.Shape[2] != dim3)
                throw BadRank("Expected shape: {0}, actual: {1}.",
                    FormatShape(dim1, dim2, dim3), FormatShape(arr.Shape));
        }

        public static void AssertOfShape<T1, T2>(this Array<T1> arr, Array<T2> shape)
        {
            arr.AssertOfShape(shape.Shape);
        }

        public static void AssertOfDim<T>(this Array<T> arr, int dim)
        {
            if (dim != arr.Shape.Length)
                throw BadRank("AssertArray. AssertOfDim failed. Dims don't match. Expected:<{0}>. Actual:<{1}>",
                    dim, arr.Shape.Length);
        }

        public static void AssertOfDimConvolution2dValid<T1, T2>(this Array<T1> arr, Array<T2> kernel)
        {
            if (((arr.Shape[0] < kernel.Shape[0]) || (arr.Shape[1] < kernel.Shape[1])) && !((arr.Shape[0] < kernel.Shape[0]) && (arr.Shape[1] < kernel.Shape[1])))
                throw Failed("Array should have at least as many items as Kernel in every dimension for 'valid' mode.");
        }

        static System.Random rnd = new System.Random();

        public static IEnumerable<Array<T>> GenerateArrays<T>(Array<T> arr, Func<int[], Array<T>> factory)
        {
            var dim = arr.Shape.Length;
            if (dim == 1) return GenerateVecs(arr, factory);
            else if (dim == 2) return GenerateMats(arr, factory);
            else throw new NotImplementedException("Only 1D and 2D arays can be generated.");
        }

        private static IEnumerable<Array<T>> GenerateVecs<T>(Array<T> vec, Func<int[], Array<T>> factory)
        {
            var dim = 1;
            vec.AssertOfDim(dim);
            var n = vec.Shape[0];

            var extraAxis = new[] { 0, 3 };
            var shapeMuls = new[] { 1, 3 };
            var shapeAdds = new[] { 0, 10 };
            var axisOffs = new[] { 0, 1, 3 };
            var offsets = new[] { 0, 5 };
            var steps = new[] { 1, -1, 2, -3 };

            foreach(int extraAxe in extraAxis)
                foreach(int axisOff in axisOffs) if(axisOff < extraAxe + dim)
                    foreach(int shapeAdd in shapeAdds)
                        foreach(int offset in offsets) if(offset <= shapeAdd)
                            foreach(int shapeMul in shapeMuls)
                                foreach (int step in steps) if(Math.Abs(step) <= shapeMul)
                                {
                                    // wrapping array creation
                                    var shape = new int[extraAxe + dim];
                                    for (int i = 0; i < shape.Length; ++i) shape[i] = rnd.Next(5) + 1;
                                    shape[axisOff] = n * shapeMul + shapeAdd;
                                    var t = factory(shape);

                                    // vector extraction
                                    var slice = new Slice[extraAxe + dim];
                                    for (int i = 0; i < shape.Length; ++i) slice[i] = rnd.Next(shape[i]);

                                    if (step > 0)
                                        slice[axisOff] = (offset..(offset + n * step), step);
                                    else
                                        slice[axisOff] = ((offset - n * step - 1)..(offset != 0 ? offset - 1 : Slicer.Start), step);

                                    // set the view with the correct values
                                    t[slice] = vec;
                                    yield return t[slice];
                                }
        }

        private static IEnumerable<Array<T>> GenerateMats<T>(Array<T> mat, Func<int[], Array<T>> factory)
        {
            var dim = 2;
            mat.AssertOfDim(dim);
            var rows = mat.Shape[0];
            var cols = mat.Shape[1];

            var extraAxis = new[] { 0, 3 };
            var shapeMuls = new[] { 1, 3 };
            var shapeAdds = new[] { 0, 10 };
            var axis1 = new[] { 0, 1, 3 };
            var axis2 = new[] { 0, 1, 4 };
            var offsets1 = new[] { 0, 5 };
            var offsets2 = new[] { 0, 2 };
            var steps1 = new[] { 1, -1, 2, -3 };
            var steps2 = new[] { -1, 2 };

            foreach (int extraAxe in extraAxis)
                foreach (int ax1 in axis1) if (ax1 < extraAxe + dim)
                foreach (int ax2 in axis2) if (ax2 < extraAxe + dim && ax2 != ax1)
                    foreach (int shapeAdd in shapeAdds)
                        foreach (int off1 in offsets1) if (off1 <= shapeAdd)
                        foreach (int off2 in offsets2) if (off2 <= shapeAdd)
                            foreach (int shapeMul in shapeMuls)
                                foreach (int step1 in steps1) if (Math.Abs(step1) <= shapeMul)
                                foreach (int step2 in steps2) if (Math.Abs(step2) <= shapeMul)
                                {
                                    // wrapping array creation
                                    var shape = new int[extraAxe + dim];
                                    for (int i = 0; i < shape.Length; ++i) shape[i] = rnd.Next(5) + 1;
                                    shape[ax1] = rows * shapeMul + shapeAdd;
                                    shape[ax2] = rows * shapeMul + shapeAdd;
                                    var t = factory(shape);

                                    // matrix extraction
                                    var slice = new Slice[extraAxe + dim];
                                    for (int i = 0; i < shape.Length; ++i) slice[i] = rnd.Next(shape[i]);

                                    if (step1 > 0)
                                        slice[ax1] = (off1..(off1 + rows * step1), step1);
                                    else
                                        slice[ax1] = ((off1 - rows * step1 - 1)..(off1 != 0 ? off1 - 1 : ^Slicer.Start), step1);
                                    if (step2 > 0)
                                        slice[ax2] = (off2..(off2 + rows * step2), step2);
                                    else
                                        slice[ax2] = ((off2 - rows * step2 - 1)..(off2 != 0 ? off2 - 1 : ^Slicer.Start), step2);

                                    // set the view with the correct values
                                    var m = ax1 > ax2 ? t[slice] : t[slice].T;
                                    m = mat;
                                    yield return m;
                                }
        }

        public static void GenerateTests<T>(Array<T> vec, Action<Array<T>> test)
        {
            GenerateTests(vec, shape => NN.Random.Uniform<T>(-5, 5, shape), test);
        }

        public static void GenerateTests<T>(Array<T> a, Func<int[], Array<T>> factory, Action<Array<T>> test)
        {
            foreach (var a1 in GenerateArrays(a, factory))
                test(a1);
        }

        public static void GenerateTests(Array<float> a, Array<float> b, Action<Array<float>, Array<float>> test)
        {
            GenerateTests(a, b, shape => NN.Random.Uniform(0, 1, shape).As<float>(), test);
        }

        public static void GenerateTests<T>(Array<T> a, Array<T> b, Func<int[], Array<T>> factory, Action<Array<T>, Array<T>> test)
        {
            foreach (var a1 in GenerateArrays(a, factory))
                foreach (var b1 in GenerateArrays(b, factory))
                    test(a1, b1);
        }

        public static void WithMessage(string message, Action a)
        {
            try { a(); }
            catch(Exception e) { throw new Exception(message, e); }
        }

        public static int TrySeveralTimes(int repeat, Action a, int allowedErrors = 0) => TrySeveralTimes(repeat, i => a(), allowedErrors);

        public static int TrySeveralTimes(int repeat, Action<int> a, int allowedErrors = 0)
        {
            int errorCount = 0;
            var errors = new List<Exception>();

            for (int _ = 0; _ < repeat; ++_)
                try { a(_); }
                catch (Exception e)
                {
                    errors.Add(e);
                    ++errorCount;
                }

            if (errorCount <= allowedErrors)
                return errorCount;
            else
                throw new AggregateException($"Failed {errorCount} times in {repeat} tries.", errors);
        }
    }
}
