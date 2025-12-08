
# import time
# from functools import wraps

# def timeit(fn):
#     @wraps(fn)
#     def _wrap(*args, **kwargs):
#         t0 = time.time()
#         out = fn(*args, **kwargs)
#         dt = (time.time() - t0) * 1000
#         print(f"[timeit] {fn.__name__}: {dt:.1f} ms")
#         return out
#     return _wrap
