# HW5 Setup

## 1. Download the dataset

From the `hw5/code/` directory:

**macOS / Linux:**

```bash
bash cs231n/datasets/get_datasets.sh
```

**Windows (PowerShell):**

> Since `curl` is not available on Windows, we use `powershell` to download and extract the dataset.

```powershell
curl.exe -O https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzf cifar-10-python.tar.gz
del cifar-10-python.tar.gz
```

You should now have a `cifar-10-batches-py/` folder inside `hw5/code/`.

## 2. Create a virtual environment

Pick whichever tool you prefer. We recommend **uv** for speed. You can also use `venv` or `conda`.
_The Python version should be >=3.10 and <=3.13. We recommend using **3.13**._

### Option A: `uv` (recommended)

> [uv](https://docs.astral.sh/uv/) is a fast, modern, and user-friendly Python toolchain.

```bash
uv venv --python 3.13
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate      # Windows
uv pip install -r requirements.txt
```

### Option B: `venv` + `pip`

> You don't need to install `venv`. It comes with Python.

```bash
python -m venv .venv --python 3.13
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### Option C: `conda`

> [conda](https://docs.anaconda.com/anaconda/install/) is a package and environment manager for Python.

```bash
conda create -n hw5 python=3.13
conda activate hw5
pip install -r requirements.txt
```

## 3. Build the fast convolution layers

The fast conv/pool layers use a Cython extension that must be compiled. This requires a C compiler.

**Install a C compiler (if you don't have one):**

| Platform              | Command                                                                                                                                   |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| macOS                 | `xcode-select --install`                                                                                                                  |
| Linux (Debian/Ubuntu) | `sudo apt install build-essential`                                                                                                        |
| Linux (Arch)          | `sudo pacman -S gcc`                                                                                                                      |
| Windows               | Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and select "Desktop development with C++" |

**Build the extension:**

> You may get some warnings during the build process. You can ignore them.

```bash
cd cs231n
python setup.py build_ext --inplace
```

## 4. Start working through the notebooks

Work through the notebooks in order:

1. `CNN-Layers.ipynb`: implement conv and pooling layers
2. `CNN-BatchNorm.ipynb`: implement spatial batch normalization
3. `CNN.ipynb`: train a CNN on CIFAR-10

## A note on Google Colab

While we recommend working locally, Colab can work. You'll need to upload the directory, download the dataset, and build the fast layers in a notebook cell before working through the notebooks.
