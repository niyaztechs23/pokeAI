# Installation Status

The supplied archive includes `ai_env/`, but it is a Windows virtual environment and is not portable to this Linux sandbox. Do not copy it into a source release.

For Windows, create a fresh Python 3.10–3.12 virtual environment, install a compatible PyTorch/TorchVision pair from the official PyTorch selector, then install the project requirements. The current requirements file is incomplete for live capture because `mss` is imported but not declared.

Initial audit command:

```powershell
python audit_project.py
```

Model commands from the original project are:

```powershell
python train.py --data_dir data --epochs 10 --batch_size 8
python predict.py --image path\to\image.png
python live_detect.py
```

These commands are not certified as working until the dependency and checkpoint verification phase completes.
