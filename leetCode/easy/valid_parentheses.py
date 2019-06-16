# Runtime: 36 ms, faster than 90.03% of Python3 online submissions for Valid Parentheses.
# Memory Usage: 13.2 MB, less than 61.23% of Python3 online submissions for Valid Parentheses.


class Solution:
    def isValid(self, s: str) -> bool:
        parens = {
            '(': ')',
            '{': '}',
            '[': ']'
            }
        stack = []
        for char in s:
            if char in parens:
                stack.append(char)
            else:
                if not stack or parens[stack.pop()] != char:
                    return False
        return False if stack else True
