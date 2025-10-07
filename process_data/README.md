# Generate 3D Rasterization Data

## process.sh

The script `process.sh` generates OpenScene v1.1 metadata from nuPlan `.db` files.  
It wraps around the `create_openscene_metadata*.py` scripts and provides a unified entry point for data processing.

### Requirements
- [NAVSim](https://github.com/autonomousvision/navsim)  
- [nuplan-devkit](https://github.com/motional/nuplan-devkit)
- [ScenarioNet](https://github.com/metadriverse/scenarionet)

### Modes
By editing the `process.sh`, you can choose different processing modes:
- **Default (`create_openscene_metadata.py`)**: Generate OpenScene metadata and rasterized multi-camera views.  
- **Perturbed (`create_openscene_metadata_perturbed.py`)**: Generate recovery-oriented trajectory perturbations (requires nuPlan logs with real camera inputs).  
- **Augmented (`create_openscene_metadata_aug.py`)**: Generate cross-agent synthesis data.  

### Usage
```bash
bash process.sh
```

