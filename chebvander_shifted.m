function C = chebvander_shifted(x, interval)
%chebyshev-vandermonde matrix with cheb points in the interval [a,b]
    a = interval(1);
    b = interval(2);
    xhat = (2*x - (a+b)) / (b-a);
    C = chebvander(xhat);
end