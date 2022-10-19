from numpy import pi

################################################################################
# BENCHMARK
################################################################################
GRAVITATIONAL_ACCELERATION = 9.81

# No topography benchmark
LENGTH = 100.0
HEIGHT_AVERAGE = 2.0
HEIGHT_AMPLITUDE = 0.1 * HEIGHT_AVERAGE
HEIGHT_WAVE_NUMBER = 3
HEIGHT_PHASE_SHIFT = 0.0
VELOCITY_AVERAGE = 1.0
VELOCITY_AMPLITUDE = 0.5
VELOCITY_WAVE_NUMBER = 1
VELOCITY_PHASE_SHIFT = pi / 2


################################################################################
# SOLVER
################################################################################
# finite element based solver
POLYNOMIAL_DEGREE = 1
ODE_SOLVER = "heun"
CFL_NUMBER = 0.1
MCL_CFL_NUMBER = 1
FLUX_APPROXIMATION = True

# finite volume based solver
GODUNOV_CFL_NUMBER = 0.5


################################################################################
# PLOT
################################################################################
PLOT_TARGET = "/home/alexey/Documents/plot.png"
PLOT_MESH_SIZE = 200

################################################################################
# ANIMATION
################################################################################
INTERVAL = 20
ANIMATION_TARGET = "/home/alexey/Documents/animation.mp4"
FRAME_FACTOR = 10  # indicates how many seconds one time unit lasts

################################################################################
# EOC
################################################################################
REFINE_NUMBER = 4
EOC_MESH_SIZE = 8

################################################################################
# Calculation
################################################################################
CALCULATION_MESH_SIZE = 400

################################################################################
# OTHERS
################################################################################
COARSENING_DEGREE = 10
JUMP_FACTOR = 30
DATA_SIZE = 20000
EPSILON = 1e-12
