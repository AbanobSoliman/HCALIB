
def calcM_bspline(k):

def TransformSE3(t):

def transformSE3(T):

def logSE3(T):

def expse3(t):

def invTrans(T):

def cumul_bspline_SE3(t, u, N):
    S = np.zeros([7, len(u) * (len(t) - N)])
    k = N + 1
    U = np.zeros([k, len(u)])
    M = calcM_bspline(k)
    for i in range(0,k):
        U[i,:] = u**i
    Bt = M@U
    zz = 0
    for i in range(0,np.shape(t,2)-k+1):
        A = TransformSE3(t[:, i])
        for pr in range(0,len(u)):
            for j in range(0,k-1):
                T1 = TransformSE3(t[:, i+j-1])
                T2 = TransformSE3(t[:, i+j])
                D = logSE3(invTrans(T1)@T2)
                A *= expse3(Bt[j+1,pr]@D)
        Aq = transformSE3(A)
        S[:, zz] = Aq
        zz += 1
        A = TransformSE3(t[:, i])
