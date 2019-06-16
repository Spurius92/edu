# https://www.youtube.com/watch?v=UiMNCXxqNpM&t=3s
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None


class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if l1 is None:
            return l2
        if l2 is None:
            return l1

        if l2.val < l1.val:
            l1, l2 = l2, l1

        current_last = l1
        p1 = l1
        p2 = l2
        p1 = p1.next

        while p1 is not None or p2 is not None:
            if p1 is None:
                current_last.next = p2
                break
            elif p2 is None:
                current_last.next = p1
                break

            if p2.val < p1.val:
                p1, p2 = p2, p1

            current_last.next = p1
            current_last = p1
            p1 = p1.next

        return l1
