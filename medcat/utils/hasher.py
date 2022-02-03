import xxhash
import dill
from io import BytesIO as StringIO


def dumps(obj, length=False):
    file = StringIO()
    dill.dump(obj, file, recurse=True)
    if length:
        return dumps(len(file.getvalue()), length=False)
    else:
        return file.getvalue()


class Hasher(object):
    def __init__(self):
        self.m = xxhash.xxh64()

    def update(self, obj, length=False):
        self.m.update(dumps(obj, length=length))

    def update_bytes(self, b):
        self.m.update(b)

    def hexdigest(self):
        return self.m.hexdigest()
