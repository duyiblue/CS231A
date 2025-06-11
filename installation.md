# Installation

This document records the first few steps we took to set up the environment, dataset, and make use of existing codebases.

I put everything in ``/iris/u/duyi/``, a folder shared by the cluster and iris workstations. However, I created the environment on workstation 15, since it is faster. Thus, we'll use iris-ws-15 to access it in the future, unless we want to replicate the environment again on the cluster.

On iris-ws-15:

```
cd /iris/u/duyi/cs231a/
git clone https://github.com/tylerlum/get_a_grip.git
```

### 1. installation

Then follow everything in https://github.com/tylerlum/get_a_grip/blob/main/docs/installation.md

a few comments:

- ``pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch`` is really slow. Just be patient with it.
- ``git checkout GetAGrip`` might not work. Just skip it.
- We **skipped the Isaac Gym section** entirely, also the last command (Environment variable that may be needed from isaacgym docs)
- We **skipped curobo section** from the command ``pip install -e . --no-build-isolation``. We'll start from this command next time.

### 2. download dataset

Then follow the "Download Dataset" section of README.md of get a grip.

```
export DOWNLOAD_URL=https://download.cs.stanford.edu/juno/get_a_grip

python get_a_grip/utils/download.py \
--download_url ${DOWNLOAD_URL} \
--include_meshdata True

export DATASET_NAME=nano

python get_a_grip/utils/download.py \
--download_url ${DOWNLOAD_URL} \
--dataset_name ${DATASET_NAME} \
--include_final_evaled_grasp_config_dicts True \
--include_nerfdata True \
--include_nerfcheckpoints True \
--include_point_clouds True

python get_a_grip/utils/download.py \
--download_url ${DOWNLOAD_URL} \
--include_pretrained_models True \
--include_fixed_sampler_grasp_config_dicts True

python get_a_grip/utils/download.py \
--download_url ${DOWNLOAD_URL} \
--include_real_world_nerfdata True \
--include_real_world_nerfcheckpoints True \
--include_real_world_point_clouds True
```

### 3. try running nerfstudio

```
cd /iris/u/duyi/cs231a/get_a_grip/data/dataset/nano/nerfdata
mkdir -p /iris/u/duyi/cs231a/torch_cache
mkdir -p /iris/u/duyi/cs231a/torch_extensions

export TORCH_HOME=/iris/u/duyi/cs231a/torch_cache
export TMPDIR=/iris/u/duyi/cs231a/tmp
export TORCH_EXTENSIONS_DIR=/iris/u/duyi/cs231a/torch_extensions

ns-train splatfacto --data ./core-mug-5c48d471200d2bf16e8a121e6886e18d_0_0622 
```

after running, you test it:

```
(get_a_grip_env) duyi@iris-ws-15:/iris/u/duyi/cs231a/get_a_grip/data/dataset/nano/nerfdata$ ns-export gaussian-splat --load-config outputs/core-mug-5c48d471200d2bf16e8a121e6886e18d_0_0622/splatfacto/2025-05-06_203108/config.yml --output-dir /iris/u/duyi/cs231a/get_a_grip/data/dataset/nano/nerfdata/splats_cup



# we can also train another sample:
cd /iris/u/duyi/cs231a/get_a_grip/data/dataset/nano/nerfdata
ns-train splatfacto --data ./sem-ToiletPaper-a83653ad2ba5562b713429629f985672_0_0827

ns-export gaussian-splat --load-config outputs/sem-ToiletPaper-a83653ad2ba5562b713429629f985672_0_0827/splatfacto/2025-05-08_162149/config.yml --output-dir /iris/u/duyi/cs231a/get_a_grip/data/dataset/nano/nerfdata/splats_toiletpaper
```

By this point, we are able to use nerfstudio to generate a .ply file (Gaussian Splatting) from images and a transformation.

### 4. a script for running nerfstudio

Write a script to run nerfstudio for multiple objects automatically. Currently, our dataset contains 2 objects, so it's good for testing this script.

```sh
#!/bin/bash

# Base directory where all objects are stored
BASE_DIR="./nerfdata"
OUTPUT_BASE_DIR="./outputs"
EXPORT_DIR="./splats"

# Loop through each subdirectory (object) in the base directory
for OBJECT_DIR in "$BASE_DIR"/*; do
    if [ -d "$OBJECT_DIR" ]; then
        OBJECT_NAME=$(basename "$OBJECT_DIR")

        # Skip folders with names shorter than 20 characters
        if [ ${#OBJECT_NAME} -lt 20 ]; then
            echo "Skipping folder: $OBJECT_NAME (name is too short)"
            continue
        fi

        echo "Starting training for: $OBJECT_NAME"
        
        # Run the training with ns-train
        # Alternate splatfacto-big to run with more gaussians
        ns-train splatfacto --vis tensorboard --data "$OBJECT_DIR"
        
        # Find the latest output folder after training
        LATEST_OUTPUT=$(ls -dt "$OUTPUT_BASE_DIR"/$OBJECT_NAME*/splatfacto/* | head -n 1)
        CONFIG_FILE="$LATEST_OUTPUT/config.yml"
        
        # Check if the config file exists before exporting
        if [ -f "$CONFIG_FILE" ]; then
            EXPORT_PATH="$EXPORT_DIR/splat_$OBJECT_NAME"
            echo "Exporting PLY for: $OBJECT_NAME"
            ns-export gaussian-splat --load-config "$CONFIG_FILE" --output-dir "$EXPORT_PATH"
            echo "Exported PLY to: $EXPORT_PATH"
        else
            echo "Config file not found for $OBJECT_NAME. Skipping export."
        fi

        echo "Finished training and exporting for: $OBJECT_NAME"
    fi
done

echo "All objects have been processed and exported!"
```

- This script is called ``generate_gaussians.sh``, and is placed in ``/iris/u/duyi/cs231a/get_a_grip/data/dataset/nano``, i.e., right inside the folder of the dataset and outside the ``nerfdata`` folder.
- We need to use command ``chmod +x generate_gaussians.sh`` before running this script. Then use command ``./generate_gaussians.sh`` to run this script.
- There will be an intermediate output generated, in a folder named ``outputs``, which is right inside the ``nano`` folder. We don't need to care about this.
- The final results (.ply file) are put in ``nano/splats/splat_{original object name}``, one folder for each object.
- Besides the object data that we want to use, there are some other random stuff in ``nano/nerfdata``. Of course, we don't want to use them; since the names of objects in this dataset are all long, we simply skip any directory whose name has fewer than 20 characters.
- ~~We need to press ctrl+c after processing each object, which is stupid but we currently don't know how to get rid of this.~~ We solved this by adding ``--vis tensorboard``.

### 5. download larger dataset

The previous data we used to test nerfstudio is just 2 samples. We want more (~100) samples. So we need to download the dataset.

```
export DOWNLOAD_URL=https://download.cs.stanford.edu/juno/get_a_grip
export DATASET_NAME=small_random

python get_a_grip/utils/download.py \
--download_url ${DOWNLOAD_URL} \
--dataset_name ${DATASET_NAME} \
--include_point_clouds True \
--include_nerfdata True \
--include_nerfcheckpoints True \
--include_fixed_sampler_grasp_config_dicts True \
--include_final_evaled_grasp_config_dicts True
```

Then there will be another dataset called ``small_random`` in parallel to ``nano``. We can also run the previous script (``generate_gaussians.sh``) in ``small_random`` to generate 100 .ply files.

Use tmux to run the script (keep it running for ten hours). Before you start, remember to:

```
export TORCH_HOME=/iris/u/duyi/cs231a/torch_cache
export TMPDIR=/iris/u/duyi/cs231a/tmp
export TORCH_EXTENSIONS_DIR=/iris/u/duyi/cs231a/torch_extensions

chmod +x generate_gaussians.sh
```

### 6. their dataset format

#### images and camera data

The images for each object is stored in `/iris/u/duyi/cs231a/get_a_grip/data/dataset/{dataset name, e.g. nano}/nerfdata/{object code, e.g. core-bottle-1a7ba1f4c892e2da30711cdbdbc73924_0_0744}/images`. There are 100 images for each object, and also a `transformation.json` provided.

#### grasps

Grasps data lays in the `final_evaled_grasp_config_dicts` directory in a dataset (e.g. `get_a_grip/data/dataset/small_random/final_evaled_grasp_config_dicts/`).

Grasps data is organized by objects. Since there are 100 objects in the "small_random" dataset, there are 100 .npy files for grasps.

Each of these .npy file is essentially a python dictionary. Each dictionary contains 8 keys:

```
['trans', 'rot', 'joint_angles', 'grasp_orientations', 'y_coll', 'y_pick', 'y_PGS', 'object_states_before_grasp']
```

Among them, `object_states_before_grasp` is their representation of the object, which we'll replace with our latent representation. `y_coll`, `y_pick`, `y_PGS` are the outputs. And the rest 4 keys represent a grasp.

The value for each key is a numpy array, and their first dimension are the same: the number of grasps for this object. This number varies across objects, typically between 400 and 1300. The shapes for all values are as follows:

```
trans: (num_grasps, 3)
rot: (num_grasps, 3, 3)
joint_angles: (num_grasps, 16)
grasp_orientations: (num_grasps, 4, 3, 3)
y_coll: (num_grasps,)
y_pick: (num_grasps,)
y_PGS: (num_grasps,)
object_states_before_grasp: (num_grasps, 6, 13)
```

For example, the shape for `trans` is (num_grasps, 3), because there are 3 coordinates for translation. The `y_coll`, `y_pick`, and `y_PGS` values for each grasp are real numbers between $[0, 1]$. They also satisfy that $y_{\text{coll}} \cdot y_{\text{pick}} = y_{\text{PGS}}$. 

In training the evaluator, we mix the grasps for all objects, obtaining a total of 74548 datapoints. (because there are 400-1300 grasps for each of the 100 objects).

### 7. Isaac Gym Installation

Download Isaac Gym from either [their official website](https://developer.nvidia.com/isaac-gym). Then:

```
cd /iris/u/duyi/cs231a/isaacgym/python
pip install -e .
```

Thentry to run an example script:

```
cd examples/
python joint_monkey.py
```

I encountered this error:

```
ImportError: libpython3.7m.so.1.0: cannot open shared object file: No such file or directory
```

I solved this by

```
export LD_LIBRARY_PATH=/iris/u/duyi/cs231n/miniconda3/envs/rlgpu/lib/
```

Now when I run the joint_monkey example, a window pops up and I can see funny white people dancing.

### 8. Our Work

For more information about the encoder, evaluator, and refiner pipeline, see README.md.
