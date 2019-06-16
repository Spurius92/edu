class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        if not nums:
            return 0

        i = 0
        j = len(nums) - 1

        while j >= 0 and nums[j] == val:
            j -= 1

        while i < j:
            if nums[i] == val:
                nums[i], nums[j] = nums[j], nums[i]

                while j >= 0 and nums[j] == val:
                    j -= 1
            i += 1
        if nums[i] == val:
            return i
        else:
            return i+1
