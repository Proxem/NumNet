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

        public static Slice Downward(Slice slice)
        {
            if (slice.Step <= 0) throw new Exception($"Positive step expected, got {slice.Step}");
            var start = slice.Range.End;
            if (start.IsFromEnd)
            {
                start = ^(start.Value + 1);
            }
            else
            {
                var value = start.Value - 1;
                if (value == -1) value = MinusOne;
                start = value;
            }
            var end = slice.Range.Start;
            if (end.IsFromEnd)
            {
                end = ^(end.Value + 1);
            }
            else
            {
                var value = end.Value - 1;
                if (value == -1) value = MinusOne;
                end = value;
            }
            return (start..end, -slice.Step);
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
