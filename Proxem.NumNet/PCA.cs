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

namespace Proxem.NumNet
{
    using static Slicer;

    public class PCA
    {
        public int nComponents;
        public Array<float> components_;
        public Array<float> means;

        public PCA(int nComponents = 0)
        {
            this.nComponents = nComponents;
        }

        public Array<float> Fit(Array<float> X, bool center = true)
        {
            var m = X.Shape[0];
            var n = X.Shape[1];
            var k = Math.Min(m, n);
            this.means = NN.Mean(X, axis: 0);
            if (center)
            {
                X -= means;
            }
            var copy = (float[])X.Values.Clone();

            var s = new float[k];
            var u = NN.Zeros<float>(m, m);
            components_ = NN.Zeros<float>(n, n);
            var superb = new float[k - 1];
            BlasNet.Lapack.gesvd('A', 'A', m, n, copy, n, s, u.Values, m, components_.Values, n, superb);
            var components = nComponents == 0 ? k : Math.Min(k, nComponents);
            SVDFlip(u, components);
            return X;
        }

        private void SVDFlip(Array<float> u, int k)
        {
            // Following svd_flip with u_based_decision from sklearn to flip columns of u :
            u = u[.., 0..Math.Min(nComponents, k)];
            var maxAbsCol = NN.Argmax(NN.Abs(u), axis: 0);
            var signs = new Array<float>(1, u.Shape[1]);
            for (int i = 0; i < u.Shape[1]; i++)
            {
                signs.Values[i] = (float)(NN.Sign(u[maxAbsCol.Values[i], i]));
            }
            u *= signs;
            components_ = components_[0..Math.Min(nComponents, k), ..];
            components_ *= signs.Reshape(u.Shape[1], 1);
        }

        public Array<float> FirstComponent(Array<float> X, bool center = true)
        {
            X = Fit(X, center);
            var fpc = components_[0, ..];
            var mpc = NN.Dot(fpc.Reshape(fpc.Shape[0], 1), fpc.Reshape(1, fpc.Shape[0]));
            return mpc;
        }

        public Array<float> Transform(Array<float> X)
        {
            AssertArray.AreEqual(X.Shape[1], components_.Shape[1]);
            X -= means;
            var reduction = NN.Dot(X, components_.T);
            X += means;
            return reduction;
        }
        public Array<float> FitTransform(Array<float> X, bool center = true)
        {
            X = Fit(X, center);
            return NN.Dot(X, components_.T);
        }

        public Array<float> InverseTransform(Array<float> X)
        {
            AssertArray.AreEqual(X.Shape[1], components_.Shape[0]);
            var XOriginal = NN.Dot(X, components_);
            if (means != null)
            {
                AssertArray.AreEqual(means.Shape[0], XOriginal.Shape[1]);
                XOriginal += means;
            }
            return XOriginal;
        }
    }
}
