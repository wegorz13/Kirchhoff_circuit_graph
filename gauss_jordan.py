import numpy as np

def gauss_jordan(A):
    n=len(A)
    rows = [i for i in range(n)]

    for c in range(n):
        for r in range(c, n):
            if abs(A[rows[r]][c])>abs(A[rows[c]][c]):
                temp = rows[r]
                rows[r] = rows[c]
                rows[c] = temp
        pivot = A[rows[c]][c]

        for i in range(c, n + 1):
            A[rows[c]][i] /= pivot

        for j in range(n):
            if rows[j]!=rows[c]:
                mul = A[rows[j]][c]
                for i in range(c, n+1):
                    A[rows[j]][i]-=mul*A[rows[c]][i]

    for r in rows:
        print(A[r])

A = [
    [2,2,-1,1,7],
     [-1,1,2,3,3],
     [3,-1,4,-1,31],
     [1,4,-2,2,2]
     ]

(gauss_jordan(A))