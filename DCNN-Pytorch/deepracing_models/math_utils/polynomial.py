import torch
import torch.nn.functional as F
from typing import Union

def polydiv(c1 : torch.Tensor, c2 : torch.Tensor):
     
    if c2[-1] == 0:
        raise ZeroDivisionError()
    # note: this is more efficient than `pu._div(polymul, c1, c2)`
    lc1 = c1.shape[0]
    lc2 = c2.shape[0]
    # print("lc1: %d" % (lc1,))
    # print("lc2: %d" % (lc2,))
    if lc1 < lc2:
        return c1[:1]*0, c1
    elif lc2 == 1:
        return c1/c2[-1], c1[:1]*0
    else:
        dlen = lc1 - lc2
        scl = c2[-1]
        c2 = c2[:-1]/scl
        i = dlen
        j = lc1 - 1
        while i >= 0:
            c1[i:j] -= c2*c1[j]
            i -= 1
            j -= 1
        return c1[j+1:]/scl, c1[:j+1]#pu.trimseq()
    

def polyproduct(a : torch.Tensor, b : torch.Tensor) -> torch.Tensor:
    if not a.shape[:-1]==b.shape[:-1]:
        raise ValueError("a and b must have same shape except in the last dimension. Got a of shape %s and b of shape %s" % (str(a.shape), str(b.shape)))           
    if b.shape[-1]>a.shape[-1]:
        return polyproduct(b,a)
    origasize = a.size()
    origbsize = b.size()

    m = origasize[-1]
    n = origbsize[-1]
    outputsize = list(origasize)
    outputsize[-1] = m+n-1 
    aflat = a.reshape(-1, m)
    bflat = b.reshape(-1, n)

    bmat = torch.stack([F.pad(bflat, (i,m-i-1), "constant", 0) for i in range(m)], dim=1)
    return torch.matmul(bmat.transpose(1,2), aflat.unsqueeze(-1)).squeeze(-1).view(outputsize)

def polycompanion(c : torch.Tensor):
    """
    Exactly the same as the NumPy functions (https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.polyroots.html) except operates on batches of polynomials.
    Return the companion matrix of c.
    
    Parameters
    ----------
    c : torch.Tensor
        N-K array of polynomial coefficients ordered from low to high
        degree. Batch dimension N, polynomial degree of K-1. 
        I.e., each c[i] is the coefficients of a polynomial: 
        c_0 + c_1*t + c_2*t^2 + ... + c_(K-1)*t^(K-1) = 0.0
    Returns
    -------
    mat : torch.Tensor
        Companion matrix of dimensions (N, deg, deg).
    Notes
    -----
    .. Docstring is a modified version of the equivalent NumPy function.
    """
    # c is a trimmed copy
    batch_size : int = int(c.shape[0])
    num_coefficients : int = int(c.shape[1])
    poly_degree : int = num_coefficients - 1
    if num_coefficients <= 2:
        raise ValueError('Series must have maximum degree of at least 3. K=1 makes no sense. Trivial case of K=2 should never call this function')
    monic_coefficients = c/(c[:,-1])[:,None]
    mat = torch.zeros( (batch_size, poly_degree, poly_degree), dtype=c.dtype, device=c.device)
    mat[:, :, -1] = -monic_coefficients[:,:-1]
    for i in range(0, mat.shape[1]-1):
        mat[:,i+1,i]=1.0
    return mat.transpose(1,2)
def linear_formula(coefs : torch.Tensor):
    b = coefs[:,0] + 0j
    m = coefs[:,1] + 0j
    roots = -b/m
    return roots.unsqueeze(-1)

def quadratic_formula(coefs : torch.Tensor):
    c = coefs[:,0] + 0j
    b = coefs[:,1] + 0j
    a = coefs[:,2] + 0j
    aeq0 = (torch.abs(a.real)<1E-9)*(torch.abs(a.imag)<1E-9)
    a[aeq0] = 1E-9 + 0j

    roots = torch.empty_like(coefs[:,0:2])
    discriminant = b**2 - 4.0*a*c

    roots[:,0] = -b + torch.sqrt(discriminant)
    roots[:,1] = -b - torch.sqrt(discriminant)
    
    return roots

def cubic_formula(coefs : torch.Tensor):
    d = coefs[:,0] + 0j
    c = coefs[:,1] + 0j
    b = coefs[:,2] + 0j
    a = coefs[:,3] + 0j
    aeq0 = (torch.abs(a.real)<1E-9)*(torch.abs(a.imag)<1E-9)
    if torch.any(aeq0):
        a[aeq0] = 1E-9 + 0j
    

    delta_0 = b**2 - 3*a*c
    delta_1 = 2*(b**3) - 9*a*b*c + 27*(a**2)*d
    one_third = torch.ones_like(coefs[0,0])/3.0
    Cvec = torch.pow( 0.5*(  delta_1   +   torch.sqrt(delta_1**2 - 4*(delta_0**3))    )  ,  one_third)
    Ceq0=(torch.abs(Cvec.real)<1E-9)*(torch.abs(Cvec.imag)<1E-9)
    if torch.any(Ceq0):
        Cvec[Cvec==0] = torch.pow( 0.5*(  delta_1[Cvec==0]    -   torch.sqrt(delta_1[Cvec==0] **2 - 4*(delta_0[Cvec==0] **3))    )  , one_third)
    Ceq0 = (torch.abs(Cvec.real)<1E-9)*(torch.abs(Cvec.imag)<1E-9)
    Cnot0 = ~Ceq0

    delta0_over_Cvec = torch.zeros_like(Cvec)
    delta0_over_Cvec[Cnot0] = delta_0[Cnot0]/Cvec[Cnot0]

    minus1_over_3a = -1.0/(3.0*a)
    
    roots = torch.empty_like(coefs[:,0:3]) + 0j
    roots[:,0] = minus1_over_3a*(b + Cvec + delta0_over_Cvec)

    one = torch.ones_like(coefs[0,0])
    Xi = (-0.5 + 0.8660254037844386j)*one
    Xisquare = Xi**2
    
    roots[:,1] = minus1_over_3a*(b + Xi*Cvec + delta0_over_Cvec/Xi)
    roots[:,2] = minus1_over_3a*(b + Xisquare*Cvec + delta0_over_Cvec/Xisquare)

    return roots

def polyroots(c):
    """
    Compute the roots of a batch of polynomials.
    Return the roots (a.k.a. "zeros") of the polynomial
    .. math:: p(x) = \\sum_i c[i] * x^i.
    Parameters
    ----------
    c : N-D torch.Tensor
    Returns
    -------
    out : torch.Tensor
        Complex-valued Tensor of the roots of the polynomials.
    Notes
    -----
    Except for the trivial case of a line (D=2, which is handled separately),
    the root estimates are obtained as the eigenvalues of the companion matrix, 
    Roots far from the origin of the complex plane may have large
    errors due to the numerical instability of the power series for such
    values. Roots with multiplicity greater than 1 will also show larger
    errors as the value of the series near such points is relatively
    insensitive to errors in the roots. Isolated roots near the origin can
    be improved by a few iterations of Newton's method.
    """
    # c is a trimmed copy
    num_coefficients : int = int(c.shape[1])
    if num_coefficients < 2:
        raise ValueError("Polynomial of degree 0 not supported.")
    if num_coefficients == 2:
        return linear_formula(c)
    elif num_coefficients == 3:
        return quadratic_formula(c)
    elif num_coefficients == 4:
        return cubic_formula(c)
    companion : torch.Tensor = polycompanion(c)
    return torch.linalg.eigvals(companion)