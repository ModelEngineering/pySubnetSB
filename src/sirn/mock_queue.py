'''Mocks the multiprocessing Queue to handle single process execution.'''


class MockQueue(object):
    # Mocks Multiprocessing Queue
    def __init__(self):
        self.queue = []

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    def put(self, item):
        self.queue.insert(0, item)

    def get(self):
        return self.queue.pop()
    
    def empty(self):
        return len(self.queue) == 0
    
    def qsize(self):
        return len(self.queue)
    
    def close(self):
        pass

    def task_done(self):
        pass

    def join(self):
        pass