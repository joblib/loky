from concurrent.futures import Future as _BaseFuture
from concurrent.futures._base import LOGGER


# To make loky._base.Future instances awaitable  by concurrent.futures.wait,
# derive our custom Future class from _BaseFuture. _invoke_callback is the only
# modification made to this class in loky.
class Future(_BaseFuture):
    def _invoke_callbacks(self):
        for callback in self._done_callbacks:
            try:
                callback(self)
            except BaseException:
                LOGGER.exception('exception calling callback for %r', self)
