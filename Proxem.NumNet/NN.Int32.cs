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
using Proxem.NumNet.Int32;

using Int = System.Int32;

namespace Proxem.NumNet
{
    /// <summary></summary>
    public static partial class NN
    {
        public static Int Sum(Array<Int> a)
        {
            return a.Sum();
        }

        public static Int Max(Array<Int> a)
        {
            return a.Max();
        }

        public static Array<Int> Max(Array<Int> a, int axis, bool keepDims = false)
        {
            return a.Max(axis, keepDims: keepDims);
        }

        public static Int Min(Array<Int> a)
        {
            return a.Min();
        }

        public static Array<Int> Min(Array<Int> a, int axis)
        {
            return a.Min(axis);
        }

        public static Int Mean(Array<Int> a)
        {
            return a.Mean();
        }

        public static int Argmax(Array<Int> a)
        {
            return a.Argmax();
        }

        public static Array<int> Argmax(Array<Int> a, int axis, Array<int> result = null)
        {
            return a.Argmax(axis, result: result);
        }
    }
}
