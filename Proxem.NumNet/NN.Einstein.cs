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
using Proxem.BlasNet;

namespace Proxem.NumNet
{
    using static EinsteinSumTools;

    public static class EinsteinSumTools
    {
        public enum EinsteinMode { INNER, ELEMENTWISE, OUTERX, OUTERY, SUMX, SUMY }

        public struct Einstein
        {
            public int? axisX, axisY, axisZ;
            public EinsteinMode mode;

            public override string ToString() => $"{axisX},{axisY}->{mode}->{axisZ}";
        }

        public static int[] EinsteinShape(int[] shapeX, int[] shapeY, Einstein[] einstein)
        {
            var ndim = einstein.Sum(e => e.axisZ == null ? 0 : 1);
            var shapeZ = new int[ndim];

            foreach (var e in einstein)
                switch (e.mode)
                {
                    case EinsteinMode.INNER:
                        AssertArray.AreEqual(shapeX[(int)e.axisX], shapeY[(int)e.axisY]);
                        if (e.axisZ != null) shapeZ[(int)e.axisZ] = 1;
                        break;
                    case EinsteinMode.ELEMENTWISE:
                        AssertArray.AreEqual(shapeX[(int)e.axisX], shapeY[(int)e.axisY]);
                        shapeZ[(int)e.axisZ] = shapeX[(int)e.axisX];
                        break;
                    case EinsteinMode.OUTERX:
                        shapeZ[(int)e.axisZ] = shapeX[(int)e.axisX];
                        break;
                    case EinsteinMode.OUTERY:
                        shapeZ[(int)e.axisZ] = shapeY[(int)e.axisY];
                        break;
                    case EinsteinMode.SUMX:
                    case EinsteinMode.SUMY:
                        if (e.axisZ != null) shapeZ[(int)e.axisZ] = 1;
                        break;
                }

            return shapeZ;
        }

        public static Tuple<string, string, string> EinsteinSplit(string einstein)
        {
            var coma = einstein.IndexOf(',');
            var arrow = einstein.IndexOf("->");

            var x = einstein.Substring(0, coma);
            var y = einstein.Substring(coma + 1, arrow - coma - 1);
            var z = einstein.Substring(arrow + 2);

            return Tuple.Create(x, y, z);
        }

        public static Einstein[] EinsteinRead(string einstein)
        {
            var xyz = EinsteinSplit(einstein);
            string x = xyz.Item1, y = xyz.Item2, z = xyz.Item3;

            var res = new List<Einstein>();
            bool[] ytreated = new bool[y.Length];
            bool[] ztreated = new bool[z.Length];

            for (int axisX = 0; axisX < x.Length; ++axisX)
            {
                var a = x[axisX];
                var axisY = y.IndexOf(a);
                if (axisY >= 0) ytreated[axisY] = true;
                var axisZ = z.IndexOf(a);
                if (axisZ >= 0) ztreated[axisZ] = true;

                if (axisY >= 0 && axisZ >= 0)
                    res.Add(Elementwise(axisX, axisY, axisZ));
                else if (axisY >= 0 && axisZ < 0)
                    res.Add(Inner(axisX, axisY));
                else if (axisY < 0 && axisZ >= 0)
                    res.Add(OuterX(axisX, axisZ));
                else
                    res.Add(SumX(axisX));
            }

            for (int axisY = 0; axisY < y.Length; ++axisY)
                if (!ytreated[axisY])
                {
                    var a = y[axisY];
                    var axisZ = z.IndexOf(a);
                    if (axisZ >= 0) ztreated[axisZ] = true;

                    if (axisZ >= 0)
                        res.Add(OuterY(axisY, axisZ));
                    else
                        res.Add(SumY(axisY));
                }

            if (!ztreated.All(_ => _)) throw new Exception($"Found unbound axis on result side in {einstein}");
            return res.ToArray();
        }

        public static Einstein Inner(int axisX, int axisY, int? axisZ = null) =>
            new Einstein { axisX = axisX, axisY = axisY, axisZ = axisZ, mode = EinsteinMode.INNER };

        public static Einstein Elementwise(int axisX, int axisY, int axisZ) =>
            new Einstein { axisX = axisX, axisY = axisY, axisZ = axisZ, mode = EinsteinMode.ELEMENTWISE };

        public static Einstein OuterX(int axisX, int axisZ) =>
            new Einstein { axisX = axisX, axisZ = axisZ, mode = EinsteinMode.OUTERX };

        public static Einstein OuterY(int axisY, int axisZ) =>
            new Einstein { axisY = axisY, axisZ = axisZ, mode = EinsteinMode.OUTERY };

        public static Einstein SumX(int axisX, int? axisZ = null) =>
            new Einstein { axisX = axisX, axisZ = axisZ, mode = EinsteinMode.SUMX };

        public static Einstein SumY(int axisY, int? axisZ = null) =>
            new Einstein { axisY = axisY, axisZ = axisZ, mode = EinsteinMode.SUMY };
    }

    public partial class NN
    {
        public static Array<float> EinsteinSum(Array<float> x, Array<float> y, string einstein, Array<float> result = null) =>
            EinsteinSum(x, y, EinsteinRead(einstein), result);

        public static Array<float> EinsteinSum(Array<float> x, Array<float> y, Einstein[] einstein, Array<float> result = null)
        {
            var resultShape = EinsteinShape(x.Shape, y.Shape, einstein);
            if (result == null) result = NN.Zeros<float>(resultShape);
            else result.AssertOfShape(resultShape);

            _EinsteinSum(0, einstein, x, x.Offset, y, y.Offset, result, result.Offset);
            return result;
        }

        private static void _EinsteinSum(int n, Einstein[] einstein,
            Array<float> x, int offX,
            Array<float> y, int offY,
            Array<float> z, int offZ
        )
        {
            if (n == einstein.Length)
            {
                z.Values[offZ] += x.Values[offX] * y.Values[offY];
                return;
            }

            var e = einstein[n];
            if (n == einstein.Length - 1)
            {
                switch (e.mode)
                {
                    case EinsteinMode.ELEMENTWISE:
                        int axisX = (int)e.axisX, axisY = (int)e.axisY, axisZ = (int)e.axisZ;
                        if (x.Stride[axisX] == 1 && y.Stride[axisY] == 1 && z.Stride[axisZ] == 1)
                        {
                            Blas.vmul(x.Shape[axisX], x.Values, offX, y.Values, offY, z.Values, offZ);
                            return;
                        }
                        break;
                    case EinsteinMode.OUTERX:
                        var axis = (int)e.axisX;
                        Blas.axpy(x.Shape[axis], y.Values[offY], x.Values, offX, x.Stride[axis], z.Values, offZ, z.Stride[(int)e.axisZ]);
                        return;
                    case EinsteinMode.OUTERY:
                        axis = (int)e.axisY;
                        Blas.axpy(y.Shape[axis], x.Values[offX], y.Values, offY, y.Stride[axis], z.Values, offZ, z.Stride[(int)e.axisZ]);
                        return;
                }
            }

                switch (e.mode)
                {
                    case EinsteinMode.INNER:
                        for (int i = 0; i < x.Shape[(int)e.axisX]; ++i)
                        {
                            _EinsteinSum(n + 1, einstein, x, offX, y, offY, z, offZ);
                            offX += x.Stride[(int)e.axisX];
                            offY += y.Stride[(int)e.axisY];
                        }
                        return;
                    case EinsteinMode.ELEMENTWISE:
                        for (int i = 0; i < x.Shape[(int)e.axisX]; ++i)
                        {
                            _EinsteinSum(n + 1, einstein, x, offX, y, offY, z, offZ);
                            offX += x.Stride[(int)e.axisX];
                            offY += y.Stride[(int)e.axisY];
                            offZ += z.Stride[(int)e.axisZ];
                        }
                        return;
                    case EinsteinMode.OUTERX:
                        var axis = (int)e.axisX;
                        for (int i = 0; i < x.Shape[axis]; ++i)
                        {
                            _EinsteinSum(n + 1, einstein, x, offX, y, offY, z, offZ);
                            offX += x.Stride[axis];
                            offZ += z.Stride[(int)e.axisZ];
                        }
                        return;
                    case EinsteinMode.OUTERY:
                        axis = (int)e.axisY;
                        for (int i = 0; i < y.Shape[axis]; ++i)
                        {
                            _EinsteinSum(n + 1, einstein, x, offX, y, offY, z, offZ);
                            offY += y.Stride[axis];
                            offZ += z.Stride[(int)e.axisZ];
                        }
                        return;
                    case EinsteinMode.SUMX:
                        axis = (int)e.axisX;
                        for (int i = 0; i < x.Shape[axis]; ++i)
                        {
                            _EinsteinSum(n + 1, einstein, x, offX, y, offY, z, offZ);
                            offX += x.Stride[axis];
                        }
                        return;
                    case EinsteinMode.SUMY:
                        axis = (int)e.axisY;
                        for (int i = 0; i < y.Shape[axis]; ++i)
                        {
                            _EinsteinSum(n + 1, einstein, x, offX, y, offY, z, offZ);
                            offY += y.Stride[axis];
                        }
                        return;
                }
        }
    }
}
