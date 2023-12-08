# SR3D
This is a repository for 3d-grid Super Resolution for H. Tanimura.

## File Structure
- `scripts/`: scripts for training and testing
- `scripts/params.py`: parameters for training and testing
- `scripts/run`: main scripts and job scripts
- `scripts/models`: model definitions (e.g. UNet)
- `scripts/layers`: layer definitions (e.g. normalization, activation)
- `scripts/blocks`: block definitions (e.g. residual blocks)
- `scripts/utils`: utility functions
- `scripts/dataloader`: dataloader definitions
- `scripts/diffusion`: diffusion model definitions (e.g. DDPM, scheduler)
- `tests/`: Jupyter notebooks for testing

## Usage
### Dependencies
- Numpy
- PyTorch
- PyTorch Lightning

### Training
```
python -m scripts.run.main --args <args> 
```
For args, see the get_parser() function in `scripts/utils/run_utils.py`.

### Testing
See `tests/SR3D.ipynb`.