# TensorFlowTOP

A highly experimental TensorFlow plugin for TouchDesigner. 

This will be updated.

## Setup 
1. Clone this repository.
2. Create a folder inside called `libs`.
3. Clone the TensorFlow repository inside of `libs`.
4. Follow the instructions below to build TensorFlow.

## Building TensorFlow from Source

1. Install `Visual Studio 2015`.
2. Install `CUDA 8.0` and `cuDNN 5.1`.
3. Install `Anaconda` for Windows.
4. Run `conda install python=3.5` to install the proper version of Python (you can
   verify that the correct version of Python is installed by opening a new `cmd`
   prompt and running the command `python`: you should see text like `Python 3.5.4 ...`
   before the interpreter is launched).
5. Install `SWIG` (latest `.exe` for Windows x64).
6. Add the `SWIG` folder location to your `PATH`, as it is referenced during the build.
7. Navigative to `libs\tensorflow\tensorflow\contrib\cmake\`.
8. Create a folder here called `build`.
9. Navigative inside this folder.
10. Run the command: `cmake .. -A x64 -DCMAKE_BUILD_TYPE=Release`: this will generate a 
   `Visual Studio 2015` solution file called `tensorflow.sln`.
11. Open this file in `Visual Studio 2015`.
12. By default, the project named `ALL_BUILD` will be selected.
13. Make sure that your configuration is set to `Release` and `x64`.
14. Right-click `ALL_BUILD` in the solution explorer and click `build`.
15. Wait about 2 hours...
16. Finally, if the build succeeds, create an **environment variable** `$TENSORFLOW_BUILD` 
    pointing to the `build` directory.

## Creating an Example C++ Project

1. Create a new `Visual Studio 2015` project.
2. Add a new file named `main.cpp`.
3. Under `C/C++ -> General -> Additional Include Directories`, add:
```
$(TENSORFLOW_BUILD);
$(TENSORFLOW_BUILD)\..\..\..\..\;
$(TENSORFLOW_BUILD)\..\..\..\..\third_party\eigen3;
$(TENSORFLOW_BUILD)\external\zlib_archive;
$(TENSORFLOW_BUILD)\external\gif_archive\giflib-5.1.4;
$(TENSORFLOW_BUILD)\external\png_archive;
$(TENSORFLOW_BUILD)\external\jpeg_archive;
$(TENSORFLOW_BUILD)\external\lmdb;
$(TENSORFLOW_BUILD)\external\eigen_archive;
$(TENSORFLOW_BUILD)\gemmlowp\src\gemmlowp;
$(TENSORFLOW_BUILD)\jsoncpp\src\jsoncpp;
$(TENSORFLOW_BUILD)\external\farmhash_archive;
$(TENSORFLOW_BUILD)\external\farmhash_archive\util;
$(TENSORFLOW_BUILD)\external\highwayhash;
$(TENSORFLOW_BUILD)\cub\src\cub;
$(TENSORFLOW_BUILD)\external\nsync\public;
$(TENSORFLOW_BUILD)\protobuf\src\protobuf\src;
$(TENSORFLOW_BUILD)\re2\install\include;
$(TENSORFLOW_BUILD)\external\sqlite;
$(TENSORFLOW_BUILD)\grpc\src\grpc\include;
$(TENSORFLOW_BUILD)\snappy\src\snappy;
```
3. Under `C/C++ -> Preprocessor -> Preprocessor Definitions`, add:
```
COMPILER_MSVC;
PLATFORM_WINDOWS;
```
4. Under `Linker -> General -> Additional Library Directories`, add `$(TENSORFLOW_BUILD)`.
5. Under `Linker -> Input -> Additional Dependencies`, add the following libraries:
```
re2\src\re2\$(Configuration)\re2.lib;
grpc\src\grpc\Release\grpc++_unsecure.lib;
grpc\src\grpc\Release\grpc_unsecure.lib;
grpc\src\grpc\Release\gpr.lib;
zlib\install\lib\zlibstatic.lib;
gif\install\lib\giflib.lib;
png\install\lib\libpng12_static.lib;
jpeg\install\lib\libjpeg.lib;
lmdb\install\lib\lmdb.lib;
jsoncpp\src\jsoncpp\src\lib_json\$(Configuration)\jsoncpp.lib;
farmhash\install\lib\farmhash.lib;
fft2d\src\lib\fft2d.lib;
highwayhash\install\lib\highwayhash.lib;
nsync\install\lib\nsync.lib;
snappy\src\snappy\Release\snappy.lib;
protobuf\src\protobuf\Release\libprotobuf.lib;
tf_cc.dir\Release\tf_cc.lib;
tf_cc_ops.dir\Release\tf_cc_ops.lib;
tf_cc_framework.dir\Release\tf_cc_framework.lib;
tf_core_cpu.dir\Release\tf_core_cpu.lib;
tf_core_direct_session.dir\Release\tf_core_direct_session.lib;
tf_core_framework.dir\Release\tf_core_framework.lib;
tf_core_kernels.dir\Release\tf_core_kernels.lib;
tf_core_lib.dir\Release\tf_core_lib.lib;
tf_core_ops.dir\Release\tf_core_ops.lib;
tf_cc_while_loop.dir\Release\tf_cc_while_loop.lib;
Release\tf_protos_cc.lib;
sqlite\install\lib\sqlite.lib
```
which are all relative to the `$(TENSORFLOW_BUILD)` path.
6. Under `Linker -> Command Line -> Additional Options`, add:
```
/machine:x64 
/ignore:4049 
/ignore:4197 
/ignore:4217 
/ignore:4221 
/WHOLEARCHIVE:tf_cc.lib
/WHOLEARCHIVE:tf_cc_framework.lib
/WHOLEARCHIVE:tf_cc_ops.lib
/WHOLEARCHIVE:tf_core_cpu.lib
/WHOLEARCHIVE:tf_core_direct_session.lib
/WHOLEARCHIVE:tf_core_framework.lib
/WHOLEARCHIVE:tf_core_kernels.lib
/WHOLEARCHIVE:tf_core_lib.lib
/WHOLEARCHIVE:tf_core_ops.lib 
/WHOLEARCHIVE:libjpeg.lib
```
to the text field
7. Back in `main.cpp`, add the following `include` directives:
```
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"
```
8. Generate the model with `export_model.py`.

# References
- `https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/cmake/README.md`
- `https://joe-antognini.github.io/machine-learning/build-windows-tf`
- `http://anonimousindonesian.blogspot.com/2017/12/tutorial-how-to-build-tensorflow.html`
- `https://github.com/jhjin/tensorflow-cpp`
- `https://gist.github.com/kyrs/9adf86366e9e4f04addb`
