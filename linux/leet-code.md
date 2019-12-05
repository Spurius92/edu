# Notes on leet code shell problems

## Tenth line problem

<https://leetcode.com/problems/tenth-line/>

We have a file `file.txt` with the following lines:

`Line 1`

`Line 2`

`Line 3`

`Line 4`

`Line 5`

`Line 6`

`Line 7`

`Line 8`

`Line 9`

`Line 10`

The task is to output to console the 10th line in the file

To do that, there are a few options:

1. `awk NR==10 file.txt`
2. `sed -n 10p file.txt`
3. `sed -ne '10p' file.txt`
4. `cat file.txt | awk NR==10`

These are simple and short solutions. I like that. Could be loop with regexes. For now i don't want to deal with it
