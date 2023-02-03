################################################################################
# BENCHMARK
################################################################################
GRAVITATIONAL_ACCELERATION = 9.81

################################################################################
# PLOT
################################################################################
PLOT_TARGET = "data/plot.png"
PLOT_MESH_SIZE = 200


################################################################################
# ANIMATION
################################################################################
ANIMATION_TARGET = "data/animation.mp4"
DURATION = 20  # how many seconds one clip should last
TIME_STEPS = 100

################################################################################
# EOC
################################################################################
REFINE_NUMBER = 4
EOC_MESH_SIZE = 8


################################################################################
# CALCULATE
################################################################################
CALCULATE_MESH_SIZE = 400


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
LOCAL_LAX_FRIEDRICHS_CFL_NUMBER = 0.4
COARSENING_DEGREE = 8

NETWORK_PATH = "data/subgrid-network.pth"
LOSS_GRAPH_PATH = "data/loss.png"
GAMMA = 1.0

################################################################################
# GENERATE-DATA
################################################################################
INPUT_RADIUS = 2
SKIP = 40  # Determines how many subgrid fluxes should be skipped
SUBGRID_FLUX_DATA_PATH = "data/data.csv"
BENCHMARK_DATA_PATH = "data/benchmark_data.csv"
OVERWRITE = True

################################################################################
# TRAIN-NETWORK
################################################################################
BATCH_SIZE = 128
LEARNING_RATE = 0.1

TRAINING_DATA_PATH = "data/train.csv"
VALIDATION_DATA_PATH = "data/validate.csv"
NETWORK_PATH = "data/subgrid-network.pth"


################################################################################
# PLOW ERROR EVOLUTION
################################################################################
ERROR_EVOLUTION_PATH = "data/error.png"
