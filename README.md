# NumNet
NumNet is an optimized library for matrix operations and scientific programming written in C\# and developed at [Proxem](https://proxem.com).
NumNet is inspired by python's [numpy](http://www.numpy.org/) library to facilitate its use by python developers.

## Table of contents

* [Requirements](#requirements)
* [Examples](#examples)
   * [Matrix creation](#vector-creation)
   * [Accessing values](accessing-values) 
   * [Base operations](#base-operations)   
* [Nuget Package](#nuget-package)
* [Contact](#contact) 
* [License](#license)

## Requirements

NumNet is developed in .Net Standard 2.0 and is compatible with both .Net Framework and .Net Core thus working on Windows and Linux platform.
For Mac OS users there shouldn't be any problem but we didn't test extensively. 

NumNet relies on **BlasNet** for the low level operations on arrays.
See [BlasNet](https://github.com/Proxem/BlasNet) documentation for further information on how to use Intel's MKL for low level operations.

## Examples

### Matrix creations

To create an empty 2-dimensional array of dimension 3 and 4 you can use 

```
var zeroArray = NN.Zeros(3, 4);
```

The following creates a 1-dimensional array with all even number from 0 to 40

```
var range = NN.Range(0, 40, step: 2);
```

To reshape the previous array to a 2-d dimension array use `var 2dRange = range.Reshape(4, 5);`.
This operation will be a O(1) if possible but it might need to copy the values 
(if the initial matrix is transposed or more generally if the data in the initial array are not contiguous.)

Random initializations are also supported, here are a few examples of the supported distributions

```
var bern = NN.Random.Bernouilli(0.5, 2, 3);  // 2 x 3 matrix
var norm = NN.Random.Normal(0, 1, 10, 10);   // 10 x 10 normally distributed matrix
var unif = NN.Random.Uniform(-1, 1, 5, 6);   // 5 x 6 uniform matrix between -1 and 1
```

### Accessing values

Let's start with a 2-d array of size (5 x 6)
```
var M = NN.Range(30).Reshape(5, 6);
```

To access a single value in the array we will use `M[i, j]`. 
NumNet also supports more complex slicing functions.
To select the first column of the array we will use

```
var vector = M[Slicer._, 0]; // 'Slicer._' correspond to ':' in numpy
```

More control on the slices is made using the following

```
var v0 = M[0]; // [0, 1, 2, 3, 4, 5]
var v1 = M[Slicer.Range(0, 3), Slicer.Upto(2)]; // [[0, 1],[6, 7],[12, 13]]
var v2 = M[1, Slicer.From(1)]; // [7, 8, 9, 10, 11]
var v3 = M[Slicer.Range(3, -1), -2]; // [16, 22]
```

### Simplifying notations

By using a static Slicer like

```
using static Slicer;
```

the above samples become

```
var vector = M[_, 0]; // '_' correspond to ':' in numpy
```

and

```
var v0 = M[0]; // [0, 1, 2, 3, 4, 5]
var v1 = M[Range(0, 3), Upto(2)]; // numpy's equivalent of M[0:3, :2]
var v2 = M[1, From(1)]; // numpy's equivalent of M[1, 1:]
var v3 = M[Range(3, -1), -2]; // numpy's equivalent of M[3:-1, -2]
```

Range(start, stop) can even be abbreviated as (start, stop):

```
var v1 = M[(0, 3), Upto(2)]; // numpy's equivalent of M[0:3, :2]
var v3 = M[(3, -1), -2]; // numpy's equivalent of M[3:-1, -2]
```

### Basic operations

The syntax for operations between multi-dimensional arrays is mostly the same as numpy (with Pascal Case).
For instance, matrix multiplications will be done with the following code

```
var M = NN.Random.Normal(0, 1, 3, 4);
var N = NN.Random.Bernouilli(0.6, 4, 5);

var MN = NN.Dot(M, N) // gives a (3 x 5) matrix
var MMNTranspose = NN.Dot(MN.T, M) // gives a (5 x 4) matrix
```

where `MN.T` stands for the transpose of `MN`.

### More examples

Please see project NumNet.Test for other examples

### Running the tests

Edit Proxem.NumNet/App.config and set "mkl:path" according to where your MKL dlls are located.
In VisualStudio, either set Test|Test settings|Default Processor Architecture to X64 or load file Test.runsettings from Test|Test settings|Select Test Settings File

If no "mkl:path" is found, a default managed BLAS provider is used. Note that this provider is slow and does not implement Lapack routines.

## Nuget Package

We provide a Nuget Package of **NumNet** to facilitate its use. It's available on [Nuget.org](https://www.nuget.org/packages/Proxem.NumNet/). 
Symbols are also available to facilitate debugging inside the package.

## Disclaimer

This is not an official Proxem product.

## Contact information

If you can't make **NumNet** work on your computer or if you have any tracks of improvement drop us an e-mail at one of the following address:
- thp@proxem.com
- joc@proxem.com

## License

NumNet is Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.
See the NOTICE file distributed with this work for additional information regarding copyright ownership.
The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
