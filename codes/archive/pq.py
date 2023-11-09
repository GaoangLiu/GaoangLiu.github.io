from bisect import bisect_left

class PriorityQueue(object):
    def __init__(self, li=[]):
        self.queue = []
        for i in li:
            self.push(i)

    def __str__(self):
        return ' '.join([str(i) for i in self.queue])

    # for checking if the queue is empty
    def isEmpty(self):
        return len(self.queue) == 0

    # for return the size of queue
    def size(self):
        return len(self.queue)

    # for inserting an element in the queue
    def push(self, data):
        insert_idx = bisect_left(self.queue, data)
        self.queue.insert(insert_idx, data)

    # for popping an element based on Priority
    def pop(self):
        return self.queue.pop()
