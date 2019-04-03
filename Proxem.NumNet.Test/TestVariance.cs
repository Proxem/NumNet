using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Proxem.NumNet.Test
{
    [TestClass]
    public class TestVariance
    {
        [TestMethod]
        public void TestStdSingle()
        {
            var a = NN.Array(new float[8] { 1, 1, 0, 1, 1, 2, 3, -1 }).Reshape(4,2);

            var stdRowBiased = a.Std(0);
            Assert.IsTrue(Math.Abs(stdRowBiased - 9.5 / 4) < 1e-3f);

            var stdRowUnbiased = a.Std(0, ddof: 1);
            Assert.IsTrue(Math.Abs(stdRowUnbiased - 9.5 / 3) < 1e-3f);

            var stdColUnbiased = a.Std(1, ddof: 1);
            Assert.IsTrue(Math.Abs(stdColUnbiased - 9) < 1e-3f);

            var stdColBiased = a.Std(1);
            Assert.IsTrue(Math.Abs(stdColBiased - 4.5) < 1e-3f);
        }

        [TestMethod]
        public void TestStdDouble()
        {
            var a = NN.Array(new double[8] { 1, 1, 0, 1, 1, 2, 3, -1 }).Reshape(4, 2);

            var stdRowBiased = a.Std(0);
            Assert.IsTrue(Math.Abs(stdRowBiased - 9.5 / 4) < 1e-3f);

            var stdRowUnbiased = a.Std(0, ddof: 1);
            Assert.IsTrue(Math.Abs(stdRowUnbiased - 9.5 / 3) < 1e-3f);

            var stdColUnbiased = a.Std(1, ddof: 1);
            Assert.IsTrue(Math.Abs(stdColUnbiased - 9) < 1e-3f);

            var stdColBiased = a.Std(1);
            Assert.IsTrue(Math.Abs(stdColBiased - 4.5) < 1e-3f);
        }
    }
}
