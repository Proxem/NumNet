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
    public static class StridedUtil
    {
        public static Action<int, T1[], int, int, T2[], int, int, R[], int, int> StoreResult<T1, T2, R>(Func<T1, T2, R> f)
        {
            return (n, a, offA, strideA, b, offB, strideB, r, offR, strideR) =>
            {
                for (int i = 0; i < n; ++i)
                {
                    r[offR] = f(a[offA], b[offB]);
                    offA += strideA;
                    offB += strideB;
                    offR += strideR;
                }
            };
        }

        public static Action<int, T1[], int, int, R[], int, int> StoreResult<T1, R>(Func<T1, R> f)
        {
            return (n, a, offA, strideA, r, offR, strideR) =>
            {
                for (int i = 0; i < n; ++i)
                {
                    r[offR] = f(a[offA]);
                    offA += strideA;
                    offR += strideR;
                }
            };
        }

        public static Action<int, R[], int, int> StoreResult<R>(Func<R> f)
        {
            return (n, r, offR, strideR) =>
            {
                for (int i = 0; i < n; ++i)
                {
                    r[offR] = f();
                    offR += strideR;
                }
            };
        }

        public static IEnumerator<T> Traverse<T>(int n, T[] a, int offA, int strideA)
        {
            for (int i = 0; i < n; ++i)
            {
                yield return a[offA];
                offA += strideA;
            }
        }

        public static Action<int, T1[], int, int, T2[], int, int, T3[], int, int> ComputeAndStore<T1, T2, T3>(Func<T1, T2, T3, T1> f)
        {
            return (n, a, offA, strideA, b, offB, strideB, c, offC, strideC) =>
            {
                for (int i = 0; i < n; ++i)
                {
                    a[offA] = f(a[offA], b[offB], c[offC]);
                    offA += strideA;
                    offB += strideB;
                    offC += strideC;
                }
            };
        }

        public static Action<int, T1[], int, int, T2[], int, int> ComputeAndStore<T1, T2>(Func<T1, T2, T1> f)
        {
            return (n, a, offA, strideA, b, offB, strideB) =>
            {
                for (int i = 0; i < n; ++i)
                {
                    a[offA] = f(a[offA], b[offB]);
                    offA += strideA;
                    offB += strideB;
                }
            };
        }
    }
}
