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
using System.Globalization;

using Real = System.Int32;

namespace Proxem.NumNet.Int32
{
    public class Operators : Operators<Real>
    {
        public override Real Convert(int i) => i;

        public override Real Parse(string s) => Real.Parse(s, CultureInfo.InvariantCulture);

        public override sealed Real Add(Real a, Real b) => a + b;

        public override sealed Real Mul(Real a, Real b) => a * b;

        public override sealed Real Sub(Real a, Real b) => a - b;

        public override sealed Real Div(Real a, Real b) => a / b;

        public override Real Neg(Real a) => -a;

        public override Real Gt(Real a, Real b) => a > b ? 1 : 0;

        public override Real GtEq(Real a, Real b) => a >= b ? 1 : 0;
    }
}