from pylab import *
from tqdm import tqdm

def find_largest_magnitude_off_diagonal(A):
    n = A.shape[0]
    A = abs(A)
    # print(A)
    maxim = convergence_threshold = 0 # if off diagonal elements are smaller than this, treat them as zero
    max_i, max_j = None, None
    for i in range(n):
        for j in range(n):
            if i==j: continue
            if maxim < A[i,j]:
                maxim = A[i,j]
                max_i = i
                max_j = j
    return max_i, max_j

def givens_rotation_matrix(n,i,j,theta):
    R = eye(n)
    R[i,i] = cos(theta)
    R[j,i] = sin(theta)
    R[i,j] = -sin(theta)
    R[j,j] = cos(theta)
    # print(R.T @ R)
    return R
    
def jacobi_evalue(A, maxiters=10**5):
    # find the largest magnitude off-diagonal entry 
    # so that we can determine the corresponding i,j and theta required 
    # for the Given's matrix. 

    assert all(A - A.T < 1e-12) # need to be symmetric
    N = A.shape[0]

    U = eye(N) # stores the eigenvectors
    
    for it in tqdm(range(maxiters)):
        # | a   b |
        # | b   c |
        i,j = find_largest_magnitude_off_diagonal(A)

        if i==j==None:
            # we reached convergence as all the off diagonal elements
            # are equal to zero
            break

        a,b,c = A[i,i], A[i,j], A[j,j]
        
        tan_2_theta = 2*b / (c - a) if c!=a else pi/2
        theta = arctan(tan_2_theta)/2

        R = givens_rotation_matrix(N, i,j, theta)

        A = R @ A @ R.T
        U = U @ R.T
        
    return A,U,it


if __name__ == "__main__":
    n = 4
    V,_ = qr(rand(n,n))
    D = diag([1,2,3,4])
    A = V @ D @ V.T
    # A = A.astype(double)
    # exit()
    
    D,U,iters = jacobi_evalue(A)
    print(D)
    # print(U-V)
    print("iters required:", iters)
