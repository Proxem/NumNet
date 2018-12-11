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

using Real = System.Double;

namespace Proxem.NumNet.Double
{
    public static class ArrayConvolutionExtension
    {
        public static Array<Real> Convolve(this Array<Real> a, Array<Real> kernel, Array<Real> result = null, ConvMode mode = ConvMode.Full)
        {
            a.AssertOfDim(1);
            kernel.AssertOfDim(1);

            int na = a.Shape[0], nk = kernel.Shape[0];
            if (nk > na) return kernel.Convolve(a, result: result, mode: mode);

            int nr, nFull = na + nk - 1;
            if (mode == ConvMode.Full) nr = nFull;
            else if (mode == ConvMode.Same) nr = na;
            else /* if (mode == Mode.Valid) */ nr = na - nk + 1;

            if (result == null)
                result = NN.Zeros<Real>(nr);

            if (mode == ConvMode.Full)
                for (int k = 0; k < nk; ++k)
                    result[(k, k + na)].Acc(a, alpha: kernel.Item[k]);

            else if (mode == ConvMode.Same)
            {
                int delta = nk - 1;
                int d0 = delta / 2;
                int d1 = delta - d0;
                for (int k = 0; k < nk; ++k)
                {
                    //for (int i = 0; i < na; ++i)
                    //    if (k + i - d0 >= 0 && k + i - d0 < na)
                    //        result[k + i - d0] += kernel[k] * a[i];
                    var iMin = Math.Max(d0 - k, 0);
                    var iMax = Math.Min(nr + d0 - k, na);
                    var r = result[(iMin + k - d0, iMax + k - d0)];
                    r.Add(a[(iMin, iMax)], alpha: kernel.Item[k], result: r);
                }
            }
            else if (mode == ConvMode.Valid)
            {
                int delta = 2 * nk - 2;
                int d0 = delta / 2;
                int d1 = delta - d0;
                for (int k = 0; k < nk; ++k)
                {
                    var iMin = Math.Max(d0 - k, 0);
                    var iMax = Math.Min(nr + d0 - k, na);
                    result.Acc(a[(iMin, iMax)], alpha: kernel.Item[k]);
                }
            }

            return result;
        }

        public static Array<Real> Correlate(this Array<Real> a, Array<Real> kernel, Array<Real> result = null, ConvMode mode = ConvMode.Valid)
        {
            a.AssertOfDim(1);
            kernel.AssertOfDim(1);

            int na = a.Shape[0], nk = kernel.Shape[0];

            int nr, nFull = na + nk - 1;
            if (mode == ConvMode.Full) nr = nFull;
            else if (mode == ConvMode.Same) nr = Math.Max(na, nk);
            else /* if (mode == Mode.Valid) */ nr = Math.Max(na, nk) - Math.Min(na, nk) + 1;

            if (result == null)
                result = NN.Zeros<Real>(nr);
            else
                result.Clear();

            if (mode == ConvMode.Full)
                for (int k = 0; k < nk; ++k)
                    result[(k, k + na)].Acc(a, alpha: kernel.Item[nk - 1 - k]);

            else if (mode == ConvMode.Same || mode == ConvMode.Valid)
            {
                int delta = nFull - nr;
                int d0 = delta / 2;
                int d1 = delta - d0;
                for (int k = 0; k < nk; ++k)
                {
                    var iMin = Math.Max(d0 - k, 0);
                    var iMax = Math.Min(nr + d0 - k, na);
                    var r = result[(iMin + k - d0, iMax + k - d0)];
                    r.Acc(a[(iMin, iMax)], alpha: kernel.Item[nk - 1 - k]);
                }
            }

            return result;
        }

        public static Array<Real> Convolve2d(this Array<Real> a, Array<Real> kernel, Array<Real> result = null, ConvMode mode = ConvMode.Full)
        {
            a.AssertOfDim(2);
            kernel.AssertOfDim(2);

            int na = a.Shape[0], nk = kernel.Shape[0];
            int ma = a.Shape[1], mk = kernel.Shape[1];

            int nr, nFull = na + nk - 1;
            int mr, mFull = ma + mk - 1;
            if (mode == ConvMode.Full)
            {
                nr = nFull;
                mr = mFull;
            }
            else if (mode == ConvMode.Same)
            {
                nr = na;
                mr = ma;
            }
            else /* if (mode == Mode.Valid) */
            {
                a.AssertOfDimConvolution2dValid(kernel);
                if (nk > na && mk > ma)
                    return kernel.Correlate2d(a, mode: mode);
                nr = na - nk + 1;
                mr = ma - mk + 1;
            }

            if (result == null)
                result = NN.Zeros<Real>(nr, mr);

            if (mode == ConvMode.Full)
            {
                for (int k = 0; k < nr; ++k)
                    for (int j = nk - 1; j >= 0; --j)
                        if (k - j >= 0 && k - j < na)
                        {
                            result[k] += a[k - j].Convolve(kernel[j], mode: ConvMode.Full);
                        }
            }
            else if (mode == ConvMode.Same)
            {
                for (int k = 0; k < nr; ++k)
                    for (int j = nk - 1; j >= 0; --j)
                        if (k - j >= 0 && k - j < na)
                            result[k] += a[k - j].Convolve(kernel[j], mode: ConvMode.Same);
            }
            else if (mode == ConvMode.Valid)
            {
                for (int k = 0; k < nr; ++k)
                    for (int j = nk - 1; j >= 0; --j)
                        if (k + nk - 1 - j >= 0 && k + nk - 1 - j < na)
                            result[k] += a[k + nk - 1 - j].Convolve(kernel[j], mode: ConvMode.Valid);
            }

            return result;
        }

        public static Array<Real> Correlate2d(this Array<Real> a, Array<Real> kernel, Array<Real> result = null, ConvMode mode = ConvMode.Valid)
        {
            a.AssertOfDim(2);
            kernel.AssertOfDim(2);

            int nk = kernel.Shape[0];
            int mk = kernel.Shape[1];
            int na = a.Shape[0];
            int ma = a.Shape[1];

            if (mode == ConvMode.Valid)
            {
                var nr = na - nk + 1;
                var mr = ma - mk + 1;
                result = NN.Zeros<Real>(nr, mr);
                int off_x = -1;
                int off_y = -1;
                for (int k = 0; k < nr; ++k)
                {
                    ++off_x;
                    for (int l = 0; l < mr; ++l)
                    {
                        ++off_y;
                        for (int i = 0; i < nk; ++i)
                            for (int j = 0; j < mk; ++j)
                                result[k, l] += a[off_x + i, off_y + j] * kernel[i, j];
                    };
                    off_y = -1;
                };
                return result;
            }
            else if (mode == ConvMode.Full)
            {
                var nr = na + nk - 1;
                var mr = ma + mk - 1;
                result = NN.Zeros<Real>(nr, mr);
                int off_x = -nk;
                int off_y = -mk;
                for (int k = 0; k < nr; ++k)
                {
                    ++off_x;
                    for (int l = 0; l < mr; ++l)
                    {
                        ++off_y;
                        for (int i = 0; i < nk; ++i)
                            for (int j = 0; j < mk; ++j)
                                if ((0 <= off_x + i) && (0 <= off_y + j))
                                {
                                    if ((off_x + i < na) && (off_y + j < ma))
                                    {
                                        result[k, l] += a[off_x + i, off_y + j] * kernel[i, j];
                                    }
                                }
                    };
                    off_y = -mk;
                };
                return result;
            }
            else
            {
                var nr = na;
                var mr = ma;
                result = NN.Zeros<Real>(nr, mr);
                int off_x = -1;
                int off_y = -1;
                for (int k = 0; k < nr; ++k)
                {
                    ++off_x;
                    for (int l = 0; l < mr; ++l)
                    {
                        ++off_y;
                        for (int i = 0; i < nk; ++i)
                            for (int j = 0; j < mk; ++j)
                                if ((0 <= off_x + i) && (0 <= off_y + j))
                                {
                                    if ((off_x + i < na) && (off_y + j < ma))
                                    {
                                        result[k, l] += a[off_x + i, off_y + j] * kernel[i, j];
                                    }
                                }
                    };
                    off_y = -1;
                };
                return result;
            }
        }
    }
}
