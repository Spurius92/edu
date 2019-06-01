class Solution:
    """
    Non-recursive solution, given by `Sashulya` on YouTube.
    
    """
    def search(self, nums: list[int], target: int) -> int:
        l = 0
        r = len(nums)
        while l + 1 < r:
            m = l + (r - l) // 2
            if nums[m] > target:
                r = m
            else:
                l = m

        return l if nums[l] == target else -1