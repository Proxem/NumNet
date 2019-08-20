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
using System.Text;

namespace Proxem.NumNet
{
    public struct Slice
    {
        public Range Range;
        public int Step;

        public const int MinusOne = int.MaxValue;   // -1 is not accepted for Range.End, use MinusOne instead

        public static Slice Downward(int step = -1)
        {
            if (step >= 0) throw new Exception($"Negative step excepted, got {step}");
            return (^1..MinusOne, step);
        }

        public Slice(Range range, int step = 1)
        {
            this.Range = range;
            this.Step = step;
        }

        public bool IsSingleton => this.Step == 0;

        public static implicit operator Slice(int i) => (i..i, 0);

        public static implicit operator Slice(Index i) => (i..i, 0);

        public static implicit operator Slice(Range s) => (s, 1);

        public static implicit operator Slice((Range range, int step) s) => new Slice(s.range, s.step);

        public override string ToString()
        {
            if (this.IsSingleton) return this.Range.Start.ToString();

            var result = new StringBuilder();
            result.Append(this.Range);
            if (this.Step != 1)
            {
                result.Append(':');
                result.Append(this.Step);
            }
            return result.ToString();
        }
    }
}
