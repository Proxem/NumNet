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
using System.Configuration;

using Microsoft.VisualStudio.TestTools.UnitTesting;

using Proxem.BlasNet;

namespace Proxem.LinearAlgebra.Test
{
    [TestClass]
    public class UnitTestInitialize
    {
        [AssemblyInitialize]
        public static void InitProvider(TestContext context)
        {
#if !NO_MKL
            var path = ConfigurationManager.AppSettings["mkl:Path"];
            if (path == null) Blas.Provider = new DefaultBlas();
            else
            {
                var threads = int.Parse(ConfigurationManager.AppSettings["mkl:Threads"]);
                var initialized = false;
                var failed = 0;
                var maxFailures = 5;
                while (!initialized && failed < maxFailures)
                {
                    try
                    {
                        StartProvider.LaunchMklRt(threads, path);
                        initialized = true;
                    }
                    catch (NotSupportedException)
                    {
                        ++failed;
                    }
                }
            }
#endif
        }
    }
}
