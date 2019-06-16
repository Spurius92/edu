class Solution:
    """
    Non-recursive solution, given by `Sashulya` on YouTube.

    """
    def search(self, nums: list[int], target: int) -> int:
        left = 0
        right = len(nums)
        while left + 1 < right:
            med = left + (right - left) // 2
            if nums[med] > target:
                right = med
            else:
                left = med

        return left if nums[left] == target else -1
