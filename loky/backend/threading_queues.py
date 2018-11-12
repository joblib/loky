import threading
from collections import deque


class Empty(Exception):
    'Exception raised by Queue.get(block=0)/get_nowait().'
    pass


class threading_SimpleQueue:
    '''Simple, unbounded FIFO queue.

    This pure Python implementation is not reentrant.
    '''
    # Note: while this pure Python version provides fairness
    # (by using a threading.Semaphore which is itself fair, being based
    #  on threading.Condition), fairness is not part of the API contract.
    # This allows the C version to use a different implementation.

    def __init__(self):
        self._queue = deque()
        self._count = threading.Semaphore(0)

    def put(self, item, block=True, timeout=None):
        '''Put the item on the queue.

        The optional 'block' and 'timeout' arguments are ignored, as this
        method never blocks.  They are provided for compatibility with the
        Queue class.
        '''
        self._queue.append(item)
        self._count.release()

    def get(self, block=True):
        '''Remove and return an item from the queue.

        If optional args 'block' is true and 'timeout' is None (the
        default), block if necessary until an item is available. If
        'timeout' is a non-negative number, it blocks at most 'timeout'
        seconds and raises the Empty exception if no item was available
        within that time.  Otherwise ('block' is false), return an item if
        one is immediately available, else raise the Empty exception
        ('timeout' is ignored in that case).
        '''
        # if timeout is not None and timeout < 0:
        #     raise ValueError("'timeout' must be a non-negative number")
        if not self._count.acquire(block):
            raise Empty
        return self._queue.popleft()

    def put_nowait(self, item):
        '''Put an item into the queue without blocking.

        This is exactly equivalent to `put(item)` and is only provided
        for compatibility with the Queue class.
        '''
        return self.put(item, block=False)

    def get_nowait(self):
        '''Remove and return an item from the queue without blocking.

        Only get an item if one is immediately available. Otherwise
        raise the Empty exception.
        '''
        return self.get(block=False)

    def empty(self):
        '''True if queue is empty, False otherwise (not reliable).'''
        return len(self._queue) == 0

    def qsize(self):
        '''Return the approximate size of the queue (not reliable!).'''
        return len(self._queue)
