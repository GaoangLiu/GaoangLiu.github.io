# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right


class Solution:
    def isValidSequence(self, root: TreeNode, arr):
        def dfs(n, i):
            if i == len(arr) - 1:
                return n and n.val == arr[i] and not n.left and not n.right

            if not n or n.val != arr[i]:
                return False
            return dfs(n.left, i + 1) or dfs(n.right, i + 1)

        return dfs(root, 0)
