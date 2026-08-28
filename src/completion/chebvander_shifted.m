function C = chebvander_shifted(x, interval)
%CHEBVANDER_SHIFTED  Chebyshev-Vandermonde matrix at the nodes X on [a, b].
    a = interval(1);
    b = interval(2);
    xhat = (2*x - (a+b)) / (b-a);
    C = chebvander(xhat);
end