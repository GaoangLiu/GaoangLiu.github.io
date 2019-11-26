import heapq

arr = []
heapq.heapify(arr)
for s in ["mobile", "mouse", "moneypot", "monitor", "mousepad"]:
    heapq.heappush(arr, s)
# heapq.heappush(arr, 'monitor')
# heapq.heappush(arr, 'moneypot')
# heapq.heappush(arr, 'monitor')

print(heapq.nsmallest(3, arr))

nums = []
heapq.heapify(nums)

for i in [4, 3, 9, 2, 5, 1, 98]:
    heapq.heappush(nums, i)

print(nums)

heap = []
# a = map(lambda item: heapq.heappush(heap, item), [1, 3, 5, 7, 9, 2, 4, 6, 8, 0])
# for item in data:
#     heapq.heappush(heap, item)

ordered = []
while heap:
    ordered.append(heapq.heappop(heap) )
print(ordered)
