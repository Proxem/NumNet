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
using Proxem.NumNet.Single;

using Real = System.Single;
using Proxem.BlasNet;

namespace Proxem.NumNet.Single
{
    public static class ArrayBias
    {
        public static readonly Slice _ = Slicer._;

        public static Array<Real> DotWithBias(this Array<Real> a, Array<Real> b, Array<Real> result = null, Real alpha = 1, Real beta = 0)
        {
            result = a[a.SlicesWithoutBias()].Dot(b, result, alpha: alpha, beta: beta);
            result.Acc(a[a.BiasSlice()], alpha);
            return result;
        }

        public static Array<Real> CombineWithBias(this Array<Real> t, Array<Real> x, Array<Real> y,
            Array<Real> result = null, Real beta = 0)
        {
            if (t.Shape.Length != 3 && x.Shape.Length != 1 && y.Shape.Length != 1)
                throw new ArgumentException();
            if (t.Shape[2] != x.Shape[0] + 1 && t.Shape[1] != y.Shape[0] + 1)
                throw new ArgumentException();

            result = t[_, Slicer.Upto(-1), Slicer.Upto(-1)].Combine21(x, y, result: result);

            // TODO check this mess
            //var biasY = t[_, Slicer.Until(-1), -1].Dot(y);
            t[_, Slicer.Upto(-1), -1].Dot(y, result: result, beta: 1); //Doesn't work actually
            //int offY = y.offset[0];
            //int offT = t.offset[0] + t.offset[1] + t.offset[2] + (t.Shape[2] - 1) * t.Stride[2];
            //for (int j = 0; j < y.Shape[0]; ++j)
            //{
            //    Blas.axpy(t.Shape[0], y.Values[offY], t.Values, offT, t.Stride[0], result.Values, result.offset[0], result.Stride[0]);
            //    offY += y.Stride[0];
            //    offT += t.Stride[1];
            //}

            //var biasX = t[_, -1, Slicer.Until(-1)].Dot(x);
            t[_, -1, Slicer.Upto(-1)].Dot(x, result: result, beta: 1);

            //var biasXY = t[_, -1, -1];
            result.Acc(t[_, -1, -1]);

            //result = result + biasX + biasY + biasXY;
            return result;
        }


        public static Slice[] SlicesWithoutBias(this Array<Real> a)
        {
            var slices = a.Slices();
            slices[slices.Length - 1] = Slicer.Upto(-1);
            return slices;
        }

        public static Slice[] BiasSlice(this Array<Real> a)
        {
            var slices = a.Slices();
            slices[slices.Length - 1] = -1;
            return slices;
        }
    }
}
