#!/usr/bin/env sh
dest_dir="/home/alexey/Dropbox/Apps/Overleaf/Master Thesis/media/"

message () {
  echo
  echo "######################################################################"
  echo "$1"
  echo "######################################################################"
}

move_error_averages () {
  message "Move error averages"
  prefix="$1"
  source_dir="$2"

  for file in "$source_dir"/*/*average*; do
    file_name=$(basename "$file")
    cp -vf "$file" "${dest_dir}/${prefix}_${file_name}"
  done
}

plot_initial_data () {
  message "Generate initial Conditions"
  ./cls1 plot swe -b b1 --initial --hide --save "${dest_dir}initial_b1.png"
  ./cls1 plot swe -b b2 --initial --hide --save "${dest_dir}initial_b2.png"
  ./cls1 plot swe -b b3 --initial --hide --save "${dest_dir}initial_b3.png"
  ./cls1 plot swe -b b4 --initial --hide --save "${dest_dir}initial_b4.png"
}

# First reduced network
generate_data_1 () {
  message "Generate Data for the first network"
  ./cls1 generate-data -d $1 -s llf
}

train_nn_1 () {
  message "Train the first model"
  ./cls1 train-network llf -e 2000 --seed 1
}

plot_model_1() {
  message "Generate plots for first reduced model"
  ./cls1 plot swe -b b1 --hide --save "${dest_dir}nn_1_b1.png" -s \
    llf +m 10000 +s Reference \
    llf +m 50 +s LLF-50 \
    llf +s LLF-400 \
    reduced-llf +s NN-LLF-50

  ./cls1 plot swe -b b2 --hide --save "${dest_dir}nn_1_b2.png" -s \
    llf +m 10000 +s Reference \
    llf +m 50 +s LLF-50 \
    llf +s LLF-400 \
    reduced-llf +s NN-LLF-50

  ./cls1 plot swe -b b3 --hide --save "${dest_dir}nn_1_b3.png" -s \
    llf +m 10000 +s Reference \
    llf +m 50 +s LLF-50 \
    llf +s LLF-400 \
    reduced-llf +s NN-LLF-50

  ./cls1 plot swe -b b4 --hide --save "${dest_dir}nn_1_b4.png" -s \
    llf +m 10000 +s Reference \
    llf +m 50 +s LLF-50 \
    llf +s LLF-400 \
    reduced-llf +s NN-LLF-50
}

plot_limited_model_1() {
  message "Generate plots for first limited reduced model"
  ./cls1 plot swe -b b1 --hide --save "${dest_dir}nn_1_mcl_b1.png" -s \
    llf +s LLF-400 \
    reduced-llf +s NN-1-50 \
    mcl +f reduced-llf +m 50 ++cfl 0.0125 +s MCL-NN-1-50

  ./cls1 plot swe -b b2 --hide --save "${dest_dir}nn_1_mcl_b2.png" -s \
    llf +s LLF-400 \
    reduced-llf +s NN-1-50 \
    mcl +f reduced-llf +m 50 ++cfl 0.0125 +s MCL-NN-1-50

  ./cls1 plot swe -b b3 --hide --save "${dest_dir}nn_1_mcl_b3.png" -s \
    llf +s LLF-400 \
    reduced-llf +s NN-1-50 \
    mcl +f reduced-llf +m 50 ++cfl 0.0125 +s MCL-NN-1-50

  ./cls1 plot swe -b b4 --hide --save "${dest_dir}nn_1_mcl_b4.png" -s \
    llf +s LLF-400 \
    reduced-llf +s NN-1-50 \
    mcl +f reduced-llf +m 50 ++cfl 0.0125 +s MCL-NN-1-50
}

parameter_variation_test_1 () {
  # Note, we use seed 2 to ensure the created benchmarks are not identical to
  # the ones created by generate-data which uses seed 1.
  message "Parameter variation test for the first model"
  ./cls1 parameter-variation-test $1 \
  --save-plot --references -v all --seed 2 \
  -s reduced-llf \
  coarse +f llf
}

cfl_variation_test_1 () {
  message "CFL variation test for the first model"
  ./cls1 parameter-variation-test $1 \
  --save-plot --references "cfl_0.05" --references-prefix "cfl_0.05"  --seed 2 \
  -s reduced-llf ++cfl 0.05 \
  coarse +f llf

  ./cls1 parameter-variation-test $1 \
  --save-plot --references "cfl_0.1" --references-prefix "cfl_0.1"  --seed 2 \
  -s reduced-llf ++cfl 0.1 \
  coarse +f llf
}

parameter_variation_test_1_lim () {
  message "Parameter variation test for the limited first model"
  ./cls1 parameter-variation-test $1 \
  --save-plot --references -v all --seed 2 \
  -s mcl +f reduced-llf +m 50 ++cfl 0.0125 \
  coarse +f llf
}

cfl_variation_test_1_lim () {
  message "CFL variation test for the limited first model"
  ./cls1 parameter-variation-test $1 \
  --save-plot --references "cfl_0.05" --references-prefix "cfl_0.05"  --seed 2 \
  -s mcl +f reduced-llf +m 50 ++cfl 0.05 \
  coarse +f llf

  ./cls1 parameter-variation-test $1 \
  --save-plot --references "cfl" --references-prefix "cfl_0.1"  --seed 2 \
  -s mcl +f reduced-llf +m 50 ++cfl 0.1 \
  coarse +f llf
}

# Third network
generate_data_2 () {
  message "Generate Data for the third network"
  ./cls1 generate-data -d data/reduced-llf-2/ -s llf +m 2000 -c 40
}

train_nn_2 () {
  message "Train the third model"
  ./cls1 train-network llf2 -e 500 --seed 2
  ./cls1 train-network llf2 -e 2000 --seed 2 --resume -p lr 0.001
}

plot_model_2() {
  message "Generate plots for third reduced model"
  ./cls1 plot swe -b b1 --hide --save "${dest_dir}nn_2_b1.png" -s \
    llf +m 10000 +s Reference \
    llf +s LLF-2000 +m 2000 \
    reduced-llf +s NN-1-50 \
    reduced-llf-2 +s NN-2-50

  ./cls1 plot swe -b b2 --hide --save "${dest_dir}nn_2_b2.png" -s \
    llf +m 10000 +s Reference \
    llf +s LLF-2000 +m 2000 \
    reduced-llf +s NN-1-50 \
    reduced-llf-2 +s NN-2-50

  ./cls1 plot swe -b b3 --hide --save "${dest_dir}nn_2_b3.png" -s \
    llf +m 10000 +s Reference \
    llf +s LLF-2000 +m 2000 \
    reduced-llf +s NN-1-50 \
    reduced-llf-2 +s NN-2-50

  ./cls1 plot swe -b b4 --hide --save "${dest_dir}nn_2_b4.png" -s \
    llf +m 10000 +s Reference \
    llf +s LLF-2000 +m 2000 \
    reduced-llf +s NN-1-50 \
    reduced-llf-2 +s NN-2-50

  ./cls1 plot swe -b b5 --hide --save "${dest_dir}nn_2_b5.png" -s \
    llf +m 10000 +s Reference \
    llf +s LLF-2000 +m 2000 \
    reduced-llf-2 +s NN-2-50
}

plot_limited_model_2() {
  message "Generate plots for third limited reduced model"
  ./cls1 plot swe -b b1 --hide --save "${dest_dir}nn_2_mcl_b1.png" -s \
    llf +s LLF-2000 +m 2000 \
    reduced-llf-2 +s NN-2-50 \
    mcl +f reduced-llf-2 +m 50 ++cfl 0.0025 +s MCL-NN-2-50

  ./cls1 plot swe -b b2 --hide --save "${dest_dir}nn_2_mcl_b2.png" -s \
    llf +s LLF-2000 +m 2000 \
    reduced-llf-2 +s NN-2-50 \
    mcl +f reduced-llf-2 +m 50 ++cfl 0.0025 +s MCL-NN-2-50

  ./cls1 plot swe -b b3 --hide --save "${dest_dir}nn_2_mcl_b3.png" -s \
    llf +s LLF-2000 +m 2000 \
    reduced-llf-2 +s NN-2-50 \
    mcl +f reduced-llf-2 +m 50 ++cfl 0.0025 +s MCL-NN-2-50

  ./cls1 plot swe -b b4 --hide --save "${dest_dir}nn_2_mcl_b4.png" -s \
    llf +s LLF-2000 +m 2000 \
    reduced-llf-2 +s NN-2-50 \
    mcl +f reduced-llf-2 +m 50 ++cfl 0.0025 +s MCL-NN-2-50

  ./cls1 plot swe -b b4 --hide --save "${dest_dir}nn_2_mcl_b4_no_nn_2.png" -s \
    llf +s LLF-2000 +m 2000 \
    mcl +f reduced-llf-2 +m 50 ++cfl 0.0025 +s MCL-NN-2-50

  ./cls1 plot swe -b b5 --hide --save "${dest_dir}nn_2_mcl_b5.png" -s \
    llf +s LLF-2000 +m 2000 \
    reduced-llf-2 +s NN-2-50 \
    mcl +f reduced-llf-2 +m 50 ++cfl 0.0025 +s MCL-NN-2-50
}

parameter_variation_test_2_lim () {
  message "Parameter variation test for the limited third model"
  ./cls1 parameter-variation-test $1 \
    --save-plot --references -v all --seed 2 \
    -s mcl +f reduced-llf-2 +m 50 ++cfl 0.0025\
    coarse +c 40 +f llf +m 2000
}

cfl_variation_test_2_lim() {
  message "CFL variation test for the limited third model"
  ./cls1 parameter-variation-test $1 \
    --save-plot --references "cfl_0.05" --references-prefix "cfl_0.05" --seed 2 \
    -s mcl +f reduced-llf-2 +m 50 ++cfl 0.05\
    coarse +c 40 +f llf +m 2000

  ./cls1 parameter-variation-test $1 \
    --save-plot --references "cfl_0.1" --references-prefix "cfl_0.1" --seed 2 \
    -s mcl +f reduced-llf-2 +m 50 ++cfl 0.1\
    coarse +c 40 +f llf +m 2000
}

# main body
plot_initial_data

message "First Model"
source_dir="data/reduced-llf/ "
source_dir_lim="data/limited-reduced-llf/ "
generate_data_1
train_nn_1
parameter_variation_test_1 $source_dir
cfl_variation_test_1 $source_dir
move_error_averages nn_1 $source_dir
parameter_variation_test_1_lim $source_dir_lim
cfl_variation_test_1_lim $source_dir_lim
move_error_averages nn_1_mcl $source_dir_lim
plot_model_1
plot_limited_model_1

message "Third Model"
source_dir="data/reduced-llf-2/ "
source_dir_lim="data/limited-reduced-llf-2/ "
generate_data_2
train_nn_2
parameter_variation_test_2_lim $source_dir_lim
cfl_variation_test_2_lim $source_dir_lim
move_error_averages nn_2_mcl $source_dir_lim
plot_model_2
plot_limited_model_2

message "Finished"
