import unittest


class AsyncMock(unittest.mock.MagicMock):
    async def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)
