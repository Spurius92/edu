# converting roman numbers into integers
class Solution:
    def romanToInt(self, s: str) -> int:
        converter = {'I': 1,
                     'V': 5, 
                     'X': 10, 
                     'L': 50,
                     'C': 100, 
                     'D': 500, 
                     'M': 1000
                     }
        result = 0
        previous = 'M'
        out = []
        for i in range(len(s)):
            current = s[i]
            if converter[current] <= converter[previous]:
                out.append(converter[current])
       
            else:
                out.append(converter[current] - 2 * converter[previous])
            previous = current
        return sum(out)
