# Runtime: 68 ms, faster than 59.93% of Python3 online submissions for Container With Most Water.
# Memory Usage: 14.6 MB, less than 19.93% of Python3 online submissions for Container With Most Water.


class Solution:
    def maxArea(self, height: List[int]) -> int:
        start = 0
        stop = len(height) - 1
        area = 0
        while start < stop:
            area = max(area, ((stop - start) * min(height[start], height[stop])))
            if height[start] < height[stop]:
                start += 1
            else:
                stop -= 1
#         for i in range(start, len(height)):
#             for j in range((i + 1), len(height)):
#                 area = max(area, ((j - i) * min(height[i], height[j])))

        return area
