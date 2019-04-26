using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Proxem.NumNet.Test
{
    [TestClass]
    public class TestTensorSum
    {
        [TestMethod]
        public void TestSum3_1()
        {
            var a = NN.Range(24).Reshape(2, 3, 4).As<float>();
            var b = NN.Range(4).As<float>();

            var expected = NN.Array(new float[,,]
            {
                {
                    { 0, 2, 4, 6 },
                    { 4, 6, 8, 10 },
                    { 8, 10, 12, 14 }
                },
                {
                    { 12, 14, 16, 18 },
                    { 16, 18, 20, 22 },
                    { 20, 22, 24, 26 }
                }
            });

            var c = a + b;
            AssertArray.AreEqual(expected.Values, c.Values);
        }

        [TestMethod]
        public void TestSumInt3_1()
        {
            var a = NN.Range(24).Reshape(2, 3, 4);
            var b = NN.Range(4);

            var expected = NN.Array(new int[,,]
            {
                {
                    { 0, 2, 4, 6 },
                    { 4, 6, 8, 10 },
                    { 8, 10, 12, 14 }
                },
                {
                    { 12, 14, 16, 18 },
                    { 16, 18, 20, 22 },
                    { 20, 22, 24, 26 }
                }
            });

            var c = a + b;
            AssertArray.AreEqual(expected.Values, c.Values);
        }

        [TestMethod]
        public void TestSum3_2()
        {
            var a = NN.Range(24).Reshape(2, 3, 4).As<float>();
            var b = NN.Range(12).Reshape(3, 4).As<float>();

            var expected = NN.Array(new float[,,]
            {
                {
                    { 0, 2, 4, 6 },
                    { 8, 10, 12, 14 },
                    { 16, 18, 20, 22 }
                },
                {
                    { 12, 14, 16, 18 },
                    { 20, 22, 24, 26 },
                    { 28, 30, 32, 34 }
                }
            });

            var c = a + b;
            AssertArray.AreEqual(expected.Values, c.Values);
        }


        [TestMethod]
        public void TestSumInt3_2()
        {
            var a = NN.Range(24).Reshape(2, 3, 4);
            var b = NN.Range(12).Reshape(3, 4);

            var expected = NN.Array(new int[,,]
            {
                {
                    { 0, 2, 4, 6 },
                    { 8, 10, 12, 14 },
                    { 16, 18, 20, 22 }
                },
                {
                    { 12, 14, 16, 18 },
                    { 20, 22, 24, 26 },
                    { 28, 30, 32, 34 }
                }
            });

            var c = a + b;
            AssertArray.AreEqual(expected.Values, c.Values);
        }

        [TestMethod]
        public void TestSum4_2()
        {
            var a = NN.Range(24).Reshape(1, 2, 3, 4).As<float>();
            var b = NN.Range(12).Reshape(3, 4).As<float>();

            var expected = NN.Array(new float[,,,]
            {
                {
                    {
                        { 0, 2, 4, 6 },
                        { 8, 10, 12, 14 },
                        { 16, 18, 20, 22 }
                    },
                    {
                        { 12, 14, 16, 18 },
                        { 20, 22, 24, 26 },
                        { 28, 30, 32, 34 }
                    }
                }
            });

            var c = a + b;
            AssertArray.AreEqual(expected.Values, c.Values);
        }

        [TestMethod]
        public void TestSumInt4_2()
        {
            var a = NN.Range(24).Reshape(1, 2, 3, 4);
            var b = NN.Range(12).Reshape(3, 4);

            var expected = NN.Array(new int[,,,]
            {
                {
                    {
                        { 0, 2, 4, 6 },
                        { 8, 10, 12, 14 },
                        { 16, 18, 20, 22 }
                    },
                    {
                        { 12, 14, 16, 18 },
                        { 20, 22, 24, 26 },
                        { 28, 30, 32, 34 }
                    }
                }
            });

            var c = a + b;
            AssertArray.AreEqual(expected.Values, c.Values);
        }
    }
}
