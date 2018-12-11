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
#if CSharp6
    public struct Slice(int start, int stop)
    {
        public int Start = start;
        public int Stop = stop;
#else
    public struct Slice
    {
        public int Start;
        public int Stop;           // MinValue => singleton, MaxValue => last dim
        public int Step;

        public Slice(int start, int stop, int step = 1)
        {
            this.Start = start;
            this.Stop = stop;
            this.Step = step;
        }

        public static implicit operator Slice(int i) => Slicer.Only(i);

        public static implicit operator Slice((int start, int stop) s) => new Slice(s.start, s.stop);

        public static implicit operator Slice((int start, int stop, int step) s) => new Slice(s.start, s.stop, s.step);

        public static implicit operator Slice((object start, int stop) s) => Slicer.Range(s.start, s.stop);

        public static implicit operator Slice((object start, int stop, int step) s) => Slicer.Range(s.start, s.stop, s.step);

        public static implicit operator Slice((int start, object stop) s) => Slicer.Range(s.start, s.stop);

        public static implicit operator Slice((int start, object stop, int step) s) => Slicer.Range(s.start, s.stop, s.step);

        public bool IsSingleton()
        {
            return this.Step == 0;
        }

        public bool IsNewAxis()
        {
            return this.Step == int.MaxValue;
        }

        public override string ToString()
        {
            if (this.IsSingleton()) return this.Start.ToString();

            var result = new StringBuilder();
            if (this.Start != 0) result.Append(this.Start);
            result.Append(':');
            if (this.Stop != int.MaxValue) result.Append(this.Stop);
            if (this.Step != 1)
            {
                result.Append(':');
                result.Append(this.Step);
            }
            return result.ToString();
        }
#endif
    }

    public static class Slicer
    {
        public static readonly Slice _ = Range(0, null);
        public static readonly Slice NewAxis = Range(0, 0, int.MaxValue);

        public static Slice Range(int start, int stop, int step = 1)
        {
            return new Slice(start, stop, step);
        }

        public static Slice Range(int start, object stop, int step = 1)
        {
            if (stop == null) return From(start, step);
            throw new Exception("invalid stop value");
        }

        public static Slice Range(object start, int stop, int step = 1)
        {
            if (start == null) return Upto(stop, step);
            throw new Exception("invalid start value");
        }

        public static Slice Only(int i)
        {
            return new Slice(i, i + 1, 0);
        }

        public static Slice Step(int step)
        {
            if (step < 0) return new Slice(-1, int.MinValue, step);
            return new Slice(0, int.MaxValue, step);
        }

        [Obsolete("Use Upto")]
        public static Slice Until(int stop, int step = 1)
        {
            if (step < 0) return new Slice(-1, stop, step);
            else return new Slice(0, stop, step);
        }

        public static Slice Upto(int stop, int step = 1)
        {
            if (step < 0) return new Slice(-1, stop, step);
            else return new Slice(0, stop, step);
        }

        public static Slice From(int start, int step = 1)
        {
            if (step < 0) return new Slice(start, int.MinValue, step);
            else return new Slice(start, int.MaxValue, step);
        }
    }
}
