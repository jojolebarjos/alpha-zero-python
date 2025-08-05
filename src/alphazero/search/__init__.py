try:
    from .cython import Search
except ImportError:
    from .reference import Search
