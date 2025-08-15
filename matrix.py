class DMatrix:
    # created specifically for easier calculation of NMatrix inverse
    def __init__(self, d, n, blocks) -> None:
        """
        Usage: 
        b = [[[5, 1], [0, 6]], [[3, 4], [2, 4]], [[9, -3], [-7, 7]], [[-1, 0], [1, 8]]]
        DM = DMatrix(b)
        DM.print()

        above matrix is diagonal block representation of the following NMatrix:
        [[5, 3, 9, -1], [1, 4, -3, 0], [0, 2, -7, 1], [6, 4, 7, 8]]
        """
        self.outer = d
        self.inner = n
        self.blocks = {}
        k = 1
        for i in blocks:
            self.blocks[k] = i
            k += 1
    
    def print(self) -> None:
        print(self.blocks)

    def inner_dim(self) -> int:
        return self.inner
    
    def outer_dim(self) -> int:
        return self.outer


class NMatrix:
    # short for nested matrix (n**2 blocks, each of dxd dimension)
    type NMatrix = NMatrix
    def __init__(self, n, d, lsts) -> None:
        """
        Usage:
        l = [[4,1,5,0], [1,3,-2,1], [-2,1,-8,1], [5,2,1,1]]
        M = NMatrix(2,4,l)
        M.print()

        The lsts param contains lists, where each list contains
        the diagonal values for the inner matrix
        """
        assert len(lsts) == n ** 2
        assert (len(lsts[0])) == d
        self.hmp = {}
        self.n, self.d = n, d
        k = 0
        for j in range(1, n + 1):
            for i in range(1, n + 1):
                if len(lsts[k]) != d:
                    raise ValueError("Dimensions incorrect")
                self.hmp[(j, i)] = lsts[k]
                k += 1

    def outer_dim(self) -> int:
        return self.n

    def inner_dim(self) -> int:
        return self.d

    def print(self) -> None:
        print(self.hmp)
    
    def indices(self) -> dict:
        return self.hmp

    def add(self, other: NMatrix) -> NMatrix:
        """
        l1 = [[4,1,5,0], [1,3,-2,1], [-2,1,-8,1], [5,2,1,1]]
        l2 = [[1,2,4,-1], [0,1,-1,-1], [2,1,1,0], [1,2,6,7]]
        M1 = NMatrix(2,4,l1)
        M2 = NMatrix(2,4,l2)
        r = M1.add(M2)
        """
        if not isinstance(other, NMatrix):
            raise TypeError("Type should be NMatrix")
        assert other.outer_dim() == self.outer_dim()
        assert other.inner_dim() == self.inner_dim()
        ds1 = self.indices()
        ds2 = other.indices()
        # ds1 -> (1,1) -> [vals1]
        # ds2 -> (1,1) -> [vals2]
        l = []
        for i in range(1, self.outer_dim() + 1):
            for j in range(1, self.outer_dim() + 1):
                tmp = []
                for k in range(self.inner_dim()):
                    tmp.append(ds1[(i,j)][k] + ds2[(i,j)][k])
                l.append(tmp)
        R = NMatrix(self.outer_dim(), self.inner_dim(), l)
        return R

    def subtract(self, other: NMatrix) -> NMatrix:
        """
        l1 = [[4,1,5,0], [1,3,-2,1], [-2,1,-8,1], [5,2,1,1]]
        l2 = [[1,2,4,-1], [0,1,-1,-1], [2,1,1,0], [1,2,6,7]]
        M1 = NMatrix(2,4,l1)
        M2 = NMatrix(2,4,l2)
        r = M1.subtract(M2)
        """
        if not isinstance(other, NMatrix):
            raise TypeError("Type should be NMatrix")
        assert other.outer_dim() == self.outer_dim()
        assert other.inner_dim() == self.inner_dim()
        ds1 = self.indices()
        ds2 = other.indices()
        # ds1 -> (1,1) -> [vals1]
        # ds2 -> (1,1) -> [vals2]
        l = []
        for i in range(1, self.outer_dim() + 1):
            for j in range(1, self.outer_dim() + 1):
                tmp = []
                for k in range(self.inner_dim()):
                    tmp.append(ds1[(i,j)][k] - ds2[(i,j)][k])
                l.append(tmp)
        
        R = NMatrix(self.outer_dim(), self.inner_dim(), l)
        return R
    
    def multiply(self, other: NMatrix) -> DMatrix:
        """
        Did not choose to use Strassen's algorithm due
        to memory overhead costs
        Usage:
        l1 = [[4,1,5,0], [1,3,-2,1], [-2,1,-8,1], [5,2,1,1]]
        l2 = [[1,2,4,-1], [0,1,2,1], [3,4,-3,5], [-3,8, 0,-5]]
        M1 = NMatrix(2,4,l1)
        M2 = NMatrix(2, 4, l2)
        r = M1.multiply(M2)
        """
        if not isinstance(other, NMatrix):
            raise TypeError("Type should be NMatrix")
        assert other.outer_dim() == self.outer_dim()
        assert other.inner_dim() == self.inner_dim()
        
        M1 = self.transform()
        M2 = other.transform()
        l = []
        for i in M1.blocks:
            from numpy import matmul
            l.append(matmul(M1.blocks[i], M2.blocks[i]))

        R = DMatrix(self.outer_dim(), self.inner_dim(), l)
        return R
    
    def __add__(self, other: NMatrix) -> NMatrix:
        return self.add(other)
    
    def __sub__(self, other: NMatrix) -> NMatrix:
        return self.subtract(other)


    def __mul__(self, other: NMatrix) -> DMatrix:
        return self.multiply(other)
    
    def transpose(self) -> NMatrix:
        """
        transpose of matrix involves transposing inner matrices
        https://en.wikipedia.org/wiki/Block_matrix#Transpose

        And the inner matrices are diagonal blocks
        """
        l = []
        for i in range(1, self.outer_dim() + 1):
            for j in range(1, self.outer_dim() + 1):
                l.append(self.hmp[(j, i)])
        R = NMatrix(self.outer_dim, self.inner_dim, l)
        return R

    def transform(self) -> DMatrix:
        """
        Convert the ndxnd matrix into a diagonal block matrix
        where there are d diagonally assored blocks, each of nxn 
        dimensions, and all other blocks have zero-values
        a list of lists will be returned
        # converted to a new type DMatrix (representing d**2 blocks of nxn dims)
        """
        d = self.inner_dim()
        n = self.outer_dim()

        matrix = []
        for i in range(0, d):
            l = []
            for j in range(1, n + 1):
                tmp = []
                for k in range(1, n + 1):
                    t = self.hmp[(j, k)][i]
                    tmp.append(t)
                l.append(tmp)
            matrix.append(l)
        # outer and inner dimensions switch
        # n,d -> d,n on purpose
        T = DMatrix(d, n, matrix)
        # params switched intentionally
        return T

    def determinant(self) -> float:
        """
        Returns determinant of matrix
        """
        def det(inner_block):
            """
            Helper function to compute determinant using LU decomposition
            Time complexity: O(n^3)
            """
            from numpy.linalg import det
            return det(inner_block)

        D : DMatrix = self.transform()
        d, n = D.outer_dim(), D.inner_dim()
        tot = 1
        # from https://en.wikipedia.org/wiki/Block_matrix#Block_diagonal_matrices
        # A = diag(A1, A2, ..., An)
        # det(A) = det(A1)*det(A2)*...det(An)
        for i in D.blocks:
            tot *= det(D.blocks[i])
        return tot

    def inverse(self) -> DMatrix:
        """
        Finds inverse of a Matrix
        """
        D = self.transform()
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is not invertible")
        d, n = D.outer_dim(), D.inner_dim()
        # use numpy for inverse in nxn blocks as its way faster than
        # custom implementation
        l = []
        for i in D.blocks:
            from numpy.linalg import inv
            inv_row = inv(D.blocks[i]).tolist()
            l.append(inv_row)
        M_inv = DMatrix(d, n, l)
        return M_inv


# l1 = [[4,1,5,0], [1,3,-2,1], [-2,1,-8,1], [5,2,1,1]]
# l2 = [[1,2,4,-1], [0,1,-1,-1], [2,1,1,0], [1,2,6,7]]
# M1 = NMatrix(2,4,l1)
# M2 = NMatrix(2,4,l2)
# r = M1 + M2
# r.print()
# r.transform()
# d = r.determinant()
# print(d)

# l1 = [[4,1,5], [1,3,-2], [-2,-8,1], [5,2,7]]
# M = NMatrix(2, 3, l1)
# M_tranformed = M.transform()
# d = M.determinant()
# print(d)
# M_inv = M.inverse()
# M.print()
# M_inv.print()

# trying with det = 0
# l2 = [[1,0],[4,0],[0,5],[4,0]]
# M = NMatrix(2,2, l2)
# i = M.inverse()
# i.print()


# b = [[[5, 1], [0, 6]], [[3, 4], [2, 4]], [[9, -3], [-7, 7]], [[-1, 0], [1, 8]]]
# DM = DMatrix(2,2,b)
# DM.print()

l1 = [[4,1,5,0], [1,3,-2,1], [-2,1,-8,1], [5,2,1,1]]
l2 = [[1,2,4,-1], [0,1,2,1], [3,4,-3,5], [-3,8, 0,-5]]
M1 = NMatrix(2,4,l1)
M2 = NMatrix(2, 4, l2)
r = M1.determinant()
print(r)

