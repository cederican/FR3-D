## Test
We provide the checkpoints in [data preparation](../docs/data_preparation.md).
You need to modify the checkpoint path for both pre-trained denoiser and discriminator in the script. 

Evaluation of FR3-D without any sampling strategy.
```
sh ./Scripts/inference_denoiser_only.sh
```

Evaluation of FR3-D with the consistency metric used during sampling.
```
sh ./Scripts/inference_sampling_wMetric.sh
```

Evaluation of FR3-D with the reassembly quality estimation regressor used during sampling.
```
sh ./Scripts/inference_sampling_wRegressor.sh
```

The inference results of pose parameter are stored in ./output/denoiser/{experiemnt_name}/inference/{inference_dir}. You can use these saved results to do visualization later.

## Visualization
We use the [BlenderToolbox](https://github.com/HTDerekLiu/BlenderToolbox) to render our results. In addition, you need make sure you download the Breaking-Bad dataset everyday subset since we use mesh to visualize. 

#### Installation
```
conda create -n blender python=3.10
source activate blender
pip install -r renderer/requirements.txt
```
#### Render
```
export PYTHONPATH=$(pwd)
python renderer/render_results experiment_name=test_everyday_wRegressor inference_dir=results  renderer.output_path=results mesh_path=../Breaking-Bad-Dataset.github.io/data/
```
You can change the rendering configuration under `config/eval.yaml`. There's a section called `renderer` that contains all the configurations for rendering.
