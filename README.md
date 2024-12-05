# Animating Dynamic 3D Scenes from a Single Image
![Demo](demo/space-shuttle.gif)

This repository hosts the Machine Visual Perception course project by Group 7 (Dantong Liu, Hetong Shen, Huadi Wang). Based on [Aminate124](https://github.com/HeliosZhao/Animate124/tree/threestudio?tab=readme-ov-file), we have re-implemented the training code in Jupyter notebooks and fine-tuned configurations to generate high-quality dynamic 3D scenes generation from a single image. The repository also includes training results and metrics evaluated on benchmark examples and personalized images.

The implementation is an extension of [threestudio](https://github.com/threestudio-project/threestudio), a unified framework for 3D content creation.

## Quickstart with Google Colab
Run `MVP124.ipynb` file for dynamic 3D scene generation.

## Training
### Installation
```sh
# clone threestudio framework
git clone https://github.com/threestudio-project/threestudio.git

cd custom
git clone git@github.com:MVP-Group7/MVP124.git
```

### Personalized Image
To run with custom image, we need to preprocess images to remove background and get depth/normal maps. 
```sh
pip install backgroundremover
cd custom/MVP124
python image_preprocess.py "load/panda-dance/image.jpg" --size 512 --border_ratio 0.0
cd ../..
```
We also need to run textual inversion to support personalized modeling in stage 3.
```sh
# Textual Inversion
gpu=0
CUSTOM_DIR="custom/MVP124"
MODEL_NAME="runwayml/stable-diffusion-v1-5"
DATA_DIR="${CUSTOM_DIR}/load/panda-dance/image.jpg" # "path-to-dir-containing-your-image"
OUTPUT_DIR="outputs-textual-run/panda-dance" # "path-to-desired-output-dir"
placeholder_token="_panda_" 
init_token="_panda_" 
echo "Placeholder Token $placeholder_token"

CUDA_VISIBLE_DEVICES=$gpu accelerate launch ${CUSTOM_DIR}/textual-inversion/textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token=$placeholder_token \
  --initializer_token=$init_token \
  --resolution=512 \
  --train_batch_size=16 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=3000 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_DIR \
  --use_augmentations \
  --only_save_embeds \
  --validation_prompt "A high-resolution DSLR image of ${placeholder_token}" \
  --enable_xformers_memory_efficient_attention \
  --mixed_precision="fp16"
```

### Three Training stages
Training needs to be resumed after each stage.
```sh
seed=0
gpu=0
exp_root_dir=outputs
DATA_DIR="panda-dance"
STATIC_PROMPT="a high resolution DSLR image of panda"
DYNAMIC_PROMPT="a panda is dancing"
CN_PROMPT="a <token> is dancing"

# --------- Stage 1 (Static Stage) --------- #
python launch.py --config custom/MVP124/configs/animate124-stage1.yaml --train --gpu $gpu \
data.image.image_path=custom/MVP124/load/${DATA_DIR}/_rgba.png \
system.prompt_processor.prompt="${STATIC_PROMPT}"

# --------- Stage 2 (Dynamic Coarse Stage) --------- #
ckpt=outputs/animate124-stage1/${STATIC_PROMPT}@LAST/ckpts/last.ckpt
python launch.py --config custom/MVP124/configs/animate124-stage2-ms.yaml --train --gpu $gpu \
data.image.image_path=custom/MVP124/load/${DATA_DIR}/_rgba.png \
system.prompt_processor.prompt="${DYNAMIC_PROMPT}" \
system.weights="$ckpt"

# --------- Stage 2 (Semantic Refinement Stage) --------- #
ckpt=outputs/animate124-stage2/${DYNAMIC_PROMPT}@LAST/ckpts/last.ckpt
python launch.py --config custom/MVP124/configs/animate124-stage3-ms.yaml --train --gpu $gpu \
data.image.image_path=custom/MVP124/load/${DATA_DIR}/_rgba.png \
system.prompt_processor.prompt="${DYNAMIC_PROMPT}" \
system.prompt_processor_cn.prompt="${CN_PROMPT}" \
system.prompt_processor_cn.learned_embeds_path=custom/MVP124/load/${DATA_DIR}/learned_embeds.bin \
system.weights="$ckpt"

```


## Directory Layout
    .
    ├── configs                 # Configuration files for three training stages
    ├── data                    # Training dataset
    ├── load                    # Source images used for benchmarks and personalized training
    ├── models                  # Pretrained models
    ├── scripts                 
    ├── systems                 # Dynamic 3D scene generation pipeline
    ├── textual-inversion
    ├── utils
    ├── MVP124.ipynb            # Jupyter notebook for content generation
    └── README.md



## Credits
This code primarily re-implemnts the [Animate124](https://github.com/HeliosZhao/Animate124/tree/threestudio?tab=readme-ov-file) extension of threestudio. We thank the authors of Animate 124 and threestudio for providing the foundational framework and resources that made this project possible.

