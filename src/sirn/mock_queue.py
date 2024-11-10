'''Mocks the multiprocessing Queue to handle single process execution.'''


class MockQueue(object):
    # Mocks Multiprocessing Queue
    def __init__(self):
        self.queue = []

    def put(self, item):
        self.queue.append(item)

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