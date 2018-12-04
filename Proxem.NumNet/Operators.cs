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
    /// <summary>Default Operators class. Overriden by Double.Operators, Int32.Operators, Single.Operators.</summary>
    public class Operators<Type>
    {
        public virtual Type Convert(int i)
        {
            throw new InvalidOperationException();
        }

        public virtual Type Parse(string s)
        {
            throw new InvalidOperationException();
        }

        public virtual Type Add(Type a, Type b)
        {
            throw new InvalidOperationException();
        }

        // Default implementation, should be overriden for float and double
        public virtual Array<Type> Add(Array<Type> a, Array<Type> b) => NN.Apply(a, b, (a_, b_) => Add(a_, b_));
        public virtual Array<Type> Add(Type a, Array<Type> b) => NN.Apply(b, b_ => Add(a, b_));
        public virtual Array<Type> Add(Array<Type> a, Type b) => NN.Apply(a, a_ => Add(a_, b));

        public virtual Type Mul(Type a, Type b)
        {
            throw new InvalidOperationException();
        }

        public virtual Array<Type> Mul(Array<Type> a, Array<Type> b) => NN.Apply(a, b, (a_, b_) => Mul(a_, b_));
        public virtual Array<Type> Mul(Type a, Array<Type> b) => NN.Apply(b, b_ => Mul(a, b_));
        public virtual Array<Type> Mul(Array<Type> a, Type b) => NN.Apply(a, a_ => Mul(a_, b));

        public virtual Type Sub(Type a, Type b)
        {
            throw new InvalidOperationException();
        }

        public virtual Array<Type> Sub(Array<Type> a, Array<Type> b) => NN.Apply(a, b, (a_, b_) => Sub(a_, b_));
        public virtual Array<Type> Sub(Type a, Array<Type> b) => NN.Apply(b, b_ => Sub(a, b_));
        public virtual Array<Type> Sub(Array<Type> a, Type b) => Add(a, Neg(b));

        public virtual Type Div(Type a, Type b)
        {
            throw new InvalidOperationException();
        }

        public virtual Array<Type> Div(Array<Type> a, Array<Type> b) => NN.Apply(a, b, (a_, b_) => Div(a_, b_));
        public virtual Array<Type> Div(Type a, Array<Type> b) => NN.Apply(b, b_ => Div(a, b_));
        public virtual Array<Type> Div(Array<Type> a, Type b) => NN.Apply(a, a_ => Div(a_, b));

        public virtual Type Neg(Type a)
        {
            throw new InvalidOperationException();
        }

        public virtual Array<Type> Neg(Array<Type> a) => Mul(Convert(-1), a);

        public virtual Type Gt(Type a, Type b)
        {
            throw new InvalidOperationException();
        }

        public virtual Array<Type> Gt(Array<Type> a, Array<Type> b) => NN.Apply(a, b, (a_, b_) => Gt(a_, b_));
        public virtual Array<Type> Gt(Type a, Array<Type> b) => NN.Apply(b, b_ => Gt(a, b_));
        public virtual Array<Type> Gt(Array<Type> a, Type b) => NN.Apply(a, a_ => Gt(a_, b));

        public virtual Type GtEq(Type a, Type b)
        {
            throw new InvalidOperationException();
        }

        public virtual Array<Type> GtEq(Array<Type> a, Array<Type> b) => NN.Apply(a, b, (a_, b_) => GtEq(a_, b_));
        public virtual Array<Type> GtEq(Type a, Array<Type> b) => NN.Apply(b, b_ => GtEq(a, b_));
        public virtual Array<Type> GtEq(Array<Type> a, Type b) => NN.Apply(a, a_ => GtEq(a_, b));
    }
}
