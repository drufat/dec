from dec.spectral import *
from numpy.testing import assert_array_almost_equal as eq
random.seed(1)

def test_transforms():
    x = random.random(11)
    eq( x,          Sinv(S(x)) )
    eq( Hinv(x),    S(Hinv(Sinv(x))) )
    eq( H(S(x)),    S(H(x)) )

    x = linspace(0, 2*pi, 13)[:-1]
    h = diff(x)[0]

    f = lambda x: sin(x)
    fp = lambda x: cos(x)

    eq( H(fp(x)),  f(x+h/2) - f(x-h/2) )
    eq( fp(x), Hinv(f(x+h/2) - f(x-h/2)) )

    eq( I(fp(x), -1, 1),  f(x+1) - f(x-1) )
    eq( I(fp(x),  0, 1),  f(x+1) - f(x) )
    eq( fp(x), Iinv(f(x+1) - f(x-1), -1, 1) )
    eq( fp(x), Iinv(f(x+1) - f(x  ),  0, 1) )

    eq( S(    f(x)), f(x+h/2) )
    eq( Sinv( f(x)), f(x-h/2) )

def test_linearity():
    for N in range(4, 10):
        h = 2*pi/N
        fk = lambda x: fourier_K(x, 0, h/2)
        fk_inv = lambda x: fourier_K_inv(x, 0, h/2)
        for i in range(10):
            a = random.rand(N)
            b = random.rand(N)
            c = fk_inv(a + i*b)
            d = fk_inv(a) + i*fk_inv(b)
            assert allclose(c, d)
            c = fk(a + i*b)
            d = fk(a) + i*fk(b)
            assert allclose(c, d)

def test_fourier_K_inv():
    random.seed(1)
    for N in range(4, 10):
        h = 2*pi/N
        fk = lambda x: fourier_K(x, 0, h/2)
        fk_inv = lambda x: fourier_K_inv(x, 0, h/2)
        a = random.rand(N)
        b = fk_inv(fk(a))
        c = fk(fk_inv(a))
        assert allclose(a, b)
        assert allclose(a, c)
        K    = to_matrix(fk, N)
        Kinv = to_matrix(fk_inv, N)
        assert allclose(K.dot(a), fk(a))
        assert allclose(Kinv.dot(a), fk_inv(a))
        assert allclose(linalg.matrix_rank(Kinv, 1e-5), N)
        assert allclose(linalg.inv(K), Kinv)
        assert allclose(linalg.inv(Kinv), K)


