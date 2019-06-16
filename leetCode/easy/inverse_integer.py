# solution for LeetCode Easy challenge: Reverse integer
# input - integer(number between -2^31 and 2^ 31
# output - integer reversed
# if overflow - return 0
# get rid of 0 at the end of the number


class Solution:
    def reverse(self, x: int) -> int:

        sign = 1
        if x < 0:
            sign = -1
            x *= -1

        x = int(str(x)[::-1])

        return 0 if x > 2 ** 31 else sign * x
