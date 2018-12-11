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
    public class Random
    {
        internal System.Random Generator = new System.Random(1236);

        public System.Random Seed(int seed)
        {
            Generator = new System.Random(seed);
            return Generator;
        }

        public double NextDouble()
        {
            return Generator.NextDouble();
        }

        private double NextUniform(double min, double max) => Generator.NextDouble() * (max - min) + min;
        public Array<double> Uniform(double min, double max, params int[] shape) => NN.Fill(() => NextUniform(min, max), shape);
        public Array<double> Uniform(double min, double max, Array<double> result) => NN.Fill(() => NextUniform(min, max), result);

        private float NextUniformF(float min, float max) => (float)NextUniform(max, min);
        public Array<float> Uniform(float min, float max, params int[] shape) => NN.Fill(() => NextUniformF(min, max), shape);
        public Array<float> Uniform(float min, float max, Array<float> result) => NN.Fill(() => NextUniformF(min, max), result);

        private T NextUniform<T>(double min, double max, Type type) => (T)Convert.ChangeType(NextUniform(max, min), type);
        public Array<T> Uniform<T>(double min, double max, params int[] shape) => NN.Fill(() => NextUniform<T>(min, max, typeof(T)), shape);
        public Array<T> Uniform<T>(double min, double max, Array<T> result) => NN.Fill(() => NextUniform<T>(min, max, typeof(T)), result);

        public Array<float> Bernoulli(float p, params int[] shape)
        {
            return Bernoulli(p, new Array<float>(shape));
        }

        public Array<double> Bernoulli(double p, params int[] shape)
        {
            return Bernoulli(p, new Array<double>(shape));
        }

        public Array<float> Bernoulli(float p, Array<float> result)
        {
            for (int i = 0; i < result.Values.Length; i++)      // TODO: use shape
            {
                result.Values[i] = Generator.NextDouble() < p ? 1 : 0;
            }
            return result;
        }

        public Array<double> Bernoulli(double p, Array<double> result)
        {
            for (int i = 0; i < result.Values.Length; i++)      // TODO: use shape
            {
                result.Values[i] = Generator.NextDouble() < p ? 1 : 0;
            }
            return result;
        }

        private double NextNormal(double mean, double std)
        {
            // these are Uniform(0,1) random doubles
            var u1 = Generator.NextDouble();
            var u2 = Generator.NextDouble();

            // random Normal(0,1) using a Box-Muller transform
            var randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);

            // random Normal(mean, stdDev^2)
            var randNormal = mean + std * randStdNormal;
            return randNormal;
        }

        public Array<double> Normal(double mean, double std, params int[] shape) => NN.Fill(() => NextNormal(mean, std), shape);
        public Array<double> Normal(double mean, double std, Array<double> result) => NN.Fill(() => NextNormal(mean, std), result);

        private float NextNormalF(float mean, float std) => (float)NextNormal(mean, std);
        public Array<float> Normal(float mean, float std, params int[] shape) => NN.Fill(() => NextNormalF(mean, std), shape);
        public Array<float> Normal(float mean, float std, Array<float> result) => NN.Fill(() => NextNormalF(mean, std), result);

        public int NextInt(int max) => Generator.Next(max);

        public int Multinomial(Array<float> distribution)
        {
            distribution.AssertOfDim(1);
            var total = distribution.Sum();
            var x = Generator.NextDouble() * total;
            var stride = distribution.Stride[0];
            var len = distribution.Shape[0];
            var v = distribution.Values;
            var r = 0f;
            int off = 0;
            for(int i = 0; i < len; ++i)
            {
                r += v[off];
                if (r > x) return i;
                off += stride;
            }
            return len - 1;
        }

        public int Multinomial(IEnumerable<float> distribution, float sum = -1)
        {
            sum = sum < 0 ? distribution.Sum() : sum;
            var x = Generator.NextDouble() * sum;

            int i = 0;
            float r = 0;
            foreach (var d in distribution)
            {
                r += d;
                if (r > x) return i;
                ++i;
            }
            return i - 1;
        }

        /// <summary>Returns a label from the given multinomial distribution.</summary>
        /// <param name="distribution">The distribution, each value must be positive and represent the probabilty of the i-th label</param>
        /// <param name="sum">The sum of all values. If less than 0, the sum will be computed.</param>
        /// <returns>An integer in [0:n[, where n is the size of the distribution.</returns>
        public int Multinomial(IEnumerable<double> distribution, double sum = -1)
        {
            sum = sum < 0 ? distribution.Sum() : sum;
            var x = Generator.NextDouble() * sum;

            int i = 0;
            double r = 0;
            foreach(var d in distribution)
            {
                r += d;
                if (r > x) return i;
                ++i;
            }
            return i - 1;
        }

        public void Shuffle<T>(T[] perm)
        {
            for (int i = 0; i < perm.Length; ++i)
            {
                int j = Generator.Next(perm.Length);
                var tmp = perm[j];
                perm[j] = perm[i];
                perm[i] = tmp;
            }
        }
    }
}
