# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

### Module 3.2 Parallel Script Output
```
Parallel loop listing for  Function _tensor_matrix_multiply, /Users/pavan/Documents/Weill Cornell/Fall 2024/Machine Learning Engineering/workspace/mod3-naraharip2017/minitorch/fast_ops.py (294) 
------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                              | 
    out: Storage,                                                                         | 
    out_shape: Shape,                                                                     | 
    out_strides: Strides,                                                                 | 
    a_storage: Storage,                                                                   | 
    a_shape: Shape,                                                                       | 
    a_strides: Strides,                                                                   | 
    b_storage: Storage,                                                                   | 
    b_shape: Shape,                                                                       | 
    b_strides: Strides,                                                                   | 
) -> None:                                                                                | 
    """NUMBA tensor matrix multiply function.                                             | 
                                                                                          | 
    Should work for any tensor shapes that broadcast as long as                           | 
                                                                                          | 
    ```                                                                                   | 
    assert a_shape[-1] == b_shape[-2]                                                     | 
    ```                                                                                   | 
                                                                                          | 
    Optimizations:                                                                        | 
                                                                                          | 
    * Outer loop in parallel                                                              | 
    * No index buffers or function calls                                                  | 
    * Inner loop should have no global writes, 1 multiply.                                | 
                                                                                          | 
                                                                                          | 
    Args:                                                                                 | 
    ----                                                                                  | 
        out (Storage): storage for `out` tensor                                           | 
        out_shape (Shape): shape for `out` tensor                                         | 
        out_strides (Strides): strides for `out` tensor                                   | 
        a_storage (Storage): storage for `a` tensor                                       | 
        a_shape (Shape): shape for `a` tensor                                             | 
        a_strides (Strides): strides for `a` tensor                                       | 
        b_storage (Storage): storage for `b` tensor                                       | 
        b_shape (Shape): shape for `b` tensor                                             | 
        b_strides (Strides): strides for `b` tensor                                       | 
                                                                                          | 
    Returns:                                                                              | 
    -------                                                                               | 
        None : Fills in `out`                                                             | 
                                                                                          | 
    """                                                                                   | 
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                | 
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                | 
                                                                                          | 
    num_batches = a_shape[0] if a_batch_stride > 0 else 1                                 | 
    num_rows_a = a_shape[-2]                                                              | 
    num_cols_a_rows_b = a_shape[-1]                                                       | 
    num_cols_b = b_shape[-1]                                                              | 
                                                                                          | 
    for batch in prange(num_batches):-----------------------------------------------------| #9
        a_start = batch * a_batch_stride if a_batch_stride > 0 else 0                     | 
        b_start = batch * b_batch_stride if b_batch_stride > 0 else 0                     | 
                                                                                          | 
        for i in range(num_rows_a):                                                       | 
            for j in range(num_cols_b):                                                   | 
                dot_product = 0.0                                                         | 
                for k in range(num_cols_a_rows_b):                                        | 
                    a_pos = a_start + i * a_strides[-2] + k * a_strides[-1]               | 
                    b_pos = b_start + k * b_strides[-2] + j * b_strides[-1]               | 
                    dot_product += a_storage[a_pos] * b_storage[b_pos]                    | 
                                                                                          | 
                out_pos = (                                                               | 
                    batch * out_strides[0] + i * out_strides[-2] + j * out_strides[-1]    | 
                )                                                                         | 
                out[out_pos] = dot_product                                                | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #9).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
```