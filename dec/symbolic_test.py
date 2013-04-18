from dec.symbolic import coords_symbolic, Product, Derivative

def test_1():
    C = coords_symbolic("xy", order=1)
    prod = Product(C)    
    #1*1=1
    assert prod([1,0,0], [1,0,0]) == [1,0,0]
    #(1+x)*(1+y)=(1+x+y)
    assert prod([1,1,0], [1,0,1]) == [1,1,1]
    #(1+x+y)*(1+y)=(1+x+2y)
    assert prod([1,1,1], [1,0,1]) == [1,1,2]
    #(x)*(y)=(0)
    assert prod([0,0,1], [0,0,1]) == [0,0,0]

def test_2():
    C = coords_symbolic("xy", order=2)
    dx = Derivative(C, 'x')
    dy = Derivative(C, 'y')
    #d_x(1+x+y+xx+xy+yy)=(1+2x+y)
    assert ( dx([1, 1, 1, 1, 1, 1]) == [1, 2, 1, 0, 0, 0] )
    #d_y(1+x+y+xx+xy+yy)=(1+x+2y)
    assert ( dy([1, 1, 1, 1, 1, 1]) == [1, 1, 2, 0, 0, 0] )
