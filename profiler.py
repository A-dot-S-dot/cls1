"""This module provides profile functionalities.

If a function is decorated with `profile` but no profile information should be
printed, please set `print_profile` to `False`.

"""
import cProfile, pstats, io

PRINT_PROFILE = 0


def profile(fnc):
    """A decorator that uses cProfile to profile a function"""

    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        fnc(*args, **kwargs)
        pr.disable()

        if PRINT_PROFILE > 0:
            s = io.StringIO()
            sortby = "cumulative"
            ps = pstats.Stats(pr, stream=s)
            ps.strip_dirs().sort_stats(sortby).print_stats(PRINT_PROFILE)
            print("\n", s.getvalue())

    return inner
