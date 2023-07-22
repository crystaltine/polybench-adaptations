### Currently completed suites:
+ fdtd-2d
+ conv-3d

### WIP
+ jacobi-2d (tests not passing, but you can run the test file and see)

# Run tests & view runtime
Test script allows custom input sizes and difficulties.

Open a terminal in the folder of the benchmark and run the `test` script.

Example:
`cd <root dir>/fdtd-2d`
`python test_fdtd_2d.py`

## Default problem sizes:
+ debug (very small)
+ mini
+ small
+ medium
+ large
+ extralarge

Note that even mini can be really slow because the original implementations are unoptimized. However,
you can set custom sizes by simply pressing 'enter' when the code prompts you for `Problem Size`.