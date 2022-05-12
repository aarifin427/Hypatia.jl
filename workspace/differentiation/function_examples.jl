# Vamos specialised polynomial:
"""
src: The "specialized Vamos polynomial" (the polynomial h_2,2(x) in four variables of degree 4, on p 17 of https://arxiv.org/pdf/1512.05878.pdf)
If expanded, this will have 14 terms, 4 variables
"""
f1(x) = x[1]^2 * x[2]^2 + 4*(x[1]+x[2]+x[3]+x[4])*(x[1]*x[2]*x[3] + x[1]*x[2]*x[4] + x[1]*x[3]*x[4] + x[2]*x[3]*x[4])
p1 = [
    1 4 4 4 4 4 4 16 4 4 4 4 4 4
    2 2 2 2 1 1 1 1  1 1 1 0 0 0
    2 1 1 0 2 2 1 1  1 0 0 2 1 1
    0 1 0 1 1 0 2 1  0 2 1 1 2 1
    0 0 1 1 0 1 0 1  2 1 2 1 1 2
]

# Polynomial W 
"""
src: The polynomial W in four variables of degree four in the proof of Lemma 6.3 on page 9 of https://arxiv.org/pdf/2112.13321.pdf
7 terms, 4 variables
"""
f2(x) = x[1]^2 * x[2]^2 + x[1]^2 * x[3]^2 + x[2]^2 * x[3]^2 + x[3]^4 - 8*x[1]*x[2]*x[3]*x[4] + 2*x[1]^2 * x[4]^2 + 2*x[2]^2 * x[4]^2
p2 = [
    1 1 1 1 -8 2 2
    2 2 0 0 1 2 0
    2 0 2 0 1 0 2
    0 2 2 4  1 0 0
    0 0 0 0 1 2 2
]