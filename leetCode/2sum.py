'''
LeetCode starter code for the first problem "2sum"
First version uses two for-loops and therefore works slow.
Second utilizes dictionary and it is much faster

 input:
    nums: list of integers
    target: single integer
    output: list of two indexes, from nums, that sum up to target
'''
from timeit import timeit


class Solution:

    def twoSum(self, nums, target) :
        self.nums = nums
        self.target = target
        
        for i in range(len(self.nums)):
            for j in range(len(self.nums)):
                if (self.nums[i] + self.nums[j] == self.target) and (i != j):
                    return [i, j]


class BetterSolution:
    
    def twoSum(self, nums, target):
        if len(nums) <= 1:
            return None
        temp_dict = {}
        for i in range(len(nums)):
            if nums[i] in temp_dict:
                return [temp_dict[nums[i]], i]
            else:
                temp_dict[target - nums[i]] = i
        

if __name__ == '__main__':
    List = [int(i) for i in input().split()]
    target = int(input())
    solution1 = Solution()
    solution2 = BetterSolution()
    print('Slow solution with loops:', solution1.twoSum(List, target))
    print('Faster solution with dictionary:', solution2.twoSum(List, target))
    
    
