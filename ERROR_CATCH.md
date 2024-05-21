## No module named 'numpy.core._multiarray_umath'
### Error
```log
miniconda3/envs/icl/lib/jvm/graalvm-189e927686-java17-22.3.0/languages/python/lib-python/3/importlib/_bootstrap.py:219: UserWarning: mkl-service package failed to import, therefore Intel(R) MKL initialization ensuring its correct out-of-the box operation under condition when Gnu OpenMP had already been loaded by Python process is not assured. Please install mkl-service package, see http://github.com/IntelPython/mkl-service
  return f(*args, **kwds)
Traceback (most recent call last):
  File "miniconda3/envs/icl/lib/jvm/graalvm-189e927686-java17-22.3.0/languages/python/lib/python3.8/site-packages/numpy/core/__init__.py", line 23, in <module>
    from . import multiarray
  File "miniconda3/envs/icl/lib/jvm/graalvm-189e927686-java17-22.3.0/languages/python/lib/python3.8/site-packages/numpy/core/multiarray.py", line 10, in <module>
    from . import overrides
  File "miniconda3/envs/icl/lib/jvm/graalvm-189e927686-java17-22.3.0/languages/python/lib/python3.8/site-packages/numpy/core/overrides.py", line 6, in <module>
    from numpy.core._multiarray_umath import (
ModuleNotFoundError: No module named 'numpy.core._multiarray_umath'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "ICL/src/test/ood_data.py", line 10, in <module>
    import matplotlib.pyplot as plt
  File "miniconda3/envs/icl/lib/python3.8/site-packages/matplotlib/__init__.py", line 104, in <module>
    import numpy
  File "miniconda3/envs/icl/lib/jvm/graalvm-189e927686-java17-22.3.0/languages/python/lib/python3.8/site-packages/numpy/__init__.py", line 141, in <module>
    from . import core
  File "miniconda3/envs/icl/lib/jvm/graalvm-189e927686-java17-22.3.0/languages/python/lib/python3.8/site-packages/numpy/core/__init__.py", line 49, in <module>
    raise ImportError(msg)
ImportError: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python3.8 from "miniconda3/envs/icl/bin/python"
  * The NumPy version is: "1.24.3"

and make sure that they are the versions you expect.
Please carefully study the documentation linked above for further help.

Original error was: No module named 'numpy.core._multiarray_umath'
```
### Sol:
