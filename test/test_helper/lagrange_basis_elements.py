from numpy.polynomial import Polynomial


# Quadratic lagrange basis elements on {[0,0.25],[0.25,0.5], [0.5,0.75], [0.75,1]}
phi1_00 = Polynomial((-3, 4))
phi1_01 = Polynomial((1, -4))

phi1_10 = Polynomial((0, 4))
phi1_11 = Polynomial((2, -4))

phi1_20 = Polynomial((-1, 4))
phi1_21 = Polynomial((3, -4))

phi1_30 = Polynomial((-2, 4))
phi1_31 = Polynomial((4, -4))

basis1 = [
    (phi1_00, phi1_01),
    (phi1_10, phi1_11),
    (phi1_20, phi1_21),
    (phi1_30, phi1_31),
]
basis1_derivative = [
    (phi1_00.deriv(), phi1_01.deriv()),
    (phi1_10.deriv(), phi1_11.deriv()),
    (phi1_20.deriv(), phi1_21.deriv()),
    (phi1_30.deriv(), phi1_31.deriv()),
]


# Quadratic lagrange basis elements on {[0,0.5], [0.5,1]}
phi2_00 = Polynomial((1, -6, 8))
phi2_01 = Polynomial((3, -10, 8))

phi2_10 = Polynomial((0, 8, -16))
phi2_11 = Polynomial((0))

phi2_20 = Polynomial((0, -2, 8))
phi2_21 = Polynomial((6, -14, 8))

phi2_30 = Polynomial((0))
phi2_31 = Polynomial((-8, 24, -16))

basis2 = [
    (phi2_00, phi2_01),
    (phi2_10, phi2_11),
    (phi2_20, phi2_21),
    (phi2_30, phi2_31),
]

basis2_derivative = [
    (phi2_00.deriv(), phi2_01.deriv()),
    (phi2_10.deriv(), phi2_11.deriv()),
    (phi2_20.deriv(), phi2_21.deriv()),
    (phi2_30.deriv(), phi2_31.deriv()),
]
