## Data preparation
We use the 
[Breaking Bad Dataset](https://breaking-bad-dataset.github.io/).
You need download data from their website and follow their data process [here](https://github.com/Breaking-Bad-Dataset/Breaking-Bad-Dataset.github.io/blob/main/README.md).

After processing the data, ensure that you have a folder named `data` with the following structure:
```
../Breaking-Bad-Dataset.github.io/
└── data
    ├── breaking_bad
    │   ├── everyday
    │   │   ├── BeerBottle
    │   │   │   ├── ...
    │   │   ├── ...
    │   ├── everyday.train.txt
    │   ├── everyday.val.txt
    │   └── ...
    └── ...
```
Only the `everyday` subset is necessary.

### Generate point cloud data
In the orginal benchmark code of Breaking Bad dataset, it needs sample point cloud from mesh in each batch which is time-consuming. We pre-processing the mesh data and generate its point cloud data and its attribute.
```
cd FR3-D/
python generate_pc_data.py +data.save_pc_data_path=data/pc_data/everyday/
```

When processing the real fractured bone data use

```
cd FR3-D/
python generate_pc_data:real.py +data.save_pc_data_path=data/pc_data_real/everyday/
```


### Matching data
The matching data is only relevant when evaluating the FR3-D framework with the additional sampling strategy wMetric.

The matching data is generated using [Jigsaw](https://github.com/Jiaxin-Lu/Jigsaw). For more details about matching data generation, please refer to the guide in our [Jigsaw_matching](../Jigsaw_matching/README.md) subfolder.

## Checkpoints
We provide the checkpoints at this [link](). Still in process.

## Structure
Finally, the overall data structure should looks like:
```
FR3-D/
├── data
│   ├── pc_data
│   ├── matching_data
└── ...
├── output
│   ├── autoencoder
│   ├── denoiser
│   ├── discriminator
│   ├── ...
└── ...
```
