################################################################################
# BENCHMARK
################################################################################
GRAVITATIONAL_ACCELERATION = 9.812

################################################################################
# CALCULATE
################################################################################
CALCULATE_MESH_SIZE = 400

################################################################################
# PLOT
################################################################################
PLOT_TARGET = "data/plot.png"
PLOT_DATA_TARGET = "data/plot.pkl"
PLOT_MESH_SIZE = 200

################################################################################
# ANIMATION
################################################################################
ANIMATE_TARGET = "data/animation.mp4"
DURATION = 20  # how many seconds one clip should last
TIME_STEPS = 100

################################################################################
# EOC
################################################################################
REFINE_NUMBER = 4
EOC_MESH_SIZE = 8

################################################################################
# SOLVER
################################################################################
# finite element based solver
POLYNOMIAL_DEGREE = 1
ODE_SOLVER = "heun"
FLUX_APPROXIMATION = True
FINITE_ELEMENT_CFL_NUMBER = 0.1

# finite volume based solver
FINITE_VOLUME_CFL_NUMBER = 0.1
COARSENING_DEGREE = 8

LOSS_GRAPH_PATH = "data/loss.png"
LIMITING_GAMMA = 1.0
ANTIDIFFUSION_GAMMA = 0.1

################################################################################
# GENERATE-DATA
################################################################################
SOLUTION_NUMBER = 100
SEED = 1
INPUT_RADIUS = 2

################################################################################
# TRAIN-NETWORK
################################################################################
EPOCHS = 500
SKIP = 1  # Determines how many subgrid fluxes should be skipped
LLF_NETWORK_PATH = "data/reduced-llf/model.pkl"
MCL_NETWORK_PATH = "data/reduced-mcl/model.pkl"

################################################################################
# PLOW ERROR EVOLUTION
################################################################################
ERROR_EVOLUTION_TARGET = "data/error.png"

################################################################################
# PARAMETER VARIATION TEST
################################################################################
INITIAL_DATA_NUMBER = 20

################################################################################
# OTHER
################################################################################
EPSILON = 1e-15
