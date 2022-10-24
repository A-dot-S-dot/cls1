from numpy import pi

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
FRAME_FACTOR = 1  # indicates how many seconds one time unit lasts


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
# Generate Data
################################################################################
LOCAL_DEGREE = 1
SKIP_STEPS = 30
SOLUTION_NUMBER = 150
TRAINING_DATA_PATH = "data/train.csv"
VALIDATION_DATA_PATH = "data/validate.csv"
BENCHMARK_PARAMETERS_PATH = "data/benchmark_parameters.csv"


################################################################################
# Train Network
################################################################################
EPOCHS = 50
BATCH_SIZE = 128
HIDDEN_NEURONS = [8, 16, 8]

# scheduler
LEARNING_RATE = 1e-2
LEARNING_RATE_UPDATE_PATIENCE = 5
LEARNING_RATE_DECREASING_FACTOR = 1 / 3

NETWORK_PATH = "network/subgrid_network.pth"

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
COARSENING_DEGREE = 10


################################################################################
# Benchmark
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
# Others
################################################################################
EPSILON = 1e-12
