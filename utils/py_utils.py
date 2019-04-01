import time
import threading


def current_time_millis():
    return int(round(time.time() * 1000))


def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m


def get_class_name(instance):
    return instance.__module__ + '.' + instance.__class__.__name__


def get_class_name_from_type(module_type):
    return str(module_type).replace('class', '').replace(' ', '').replace("'", '').replace('<', '').replace('>', '')


def synchronized(func):
    func.__lock__ = threading.RLock()

    def synced_func(*args, **kws):
        with func.__lock__:
            return func(*args, **kws)

    return synced_func


def synchronized_method(method):
    outer_lock = threading.RLock()
    lock_name = "__" + method.__name__ + "_lock" + "__"

    def sync_method(self, *args, **kws):
        with outer_lock:
            if not hasattr(self, lock_name): setattr(self, lock_name, threading.Lock())
            lock = getattr(self, lock_name)
            with lock:
                return method(self, *args, **kws)

    return sync_method


class Singleton(object):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls._instance, cls):
            cls._instance = object.__new__(cls, *args, **kwargs)
        return cls._instance
