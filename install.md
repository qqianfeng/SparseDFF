## Issue of CUDA version conflict for pytorch3d

To install **CUDA 11.7** alongside an existing **CUDA 12.4** on Ubuntu, follow these steps to manage multiple CUDA versions without conflicts. This setup allows you to switch between CUDA versions as needed.

### Step-by-Step Guide:

#### 1. **Download CUDA 11.7 Installer**

* Visit the [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive) and select CUDA 11.7.
* Choose the correct installation method for your system (deb, runfile, etc.). Below, I'll provide the steps for using the **deb (network)** installer, as it is commonly used for Ubuntu.

Run the following commands to add the NVIDIA repository and install CUDA 11.7:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$(lsb_release -sr | sed 's/\.//')/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-11-7
```

This will install **CUDA 11.7** without overwriting the existing **CUDA 12.4** installation.

#### 2. **Manage Multiple CUDA Versions**

* CUDA installs to different folders by default. You can find them in `/usr/local/`. Typically, you'll see:

```bash
/usr/local/cuda-11.7
/usr/local/cuda-12.4
```

To manage different CUDA versions, you need to update your environment variables to point to the correct version as needed.

#### 3. **Set Environment Variables for CUDA**

You can switch between different CUDA versions by changing your environment variables (`PATH` and `LD_LIBRARY_PATH`). To do this temporarily, use the following commands:

* For  **CUDA 11.7** :
```bash
  export PATH=/usr/local/cuda-11.7/bin:$PATH
  export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH
```
* For  **CUDA 12.4** :
```bash
  export PATH=/usr/local/cuda-12.4/bin:$PATH
  export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
```

## Issue of import error
ImportError: cannot import name '_C' from 'pytorch3d' (/home/qian.feng/workspace/SparseDFF/thirdparty_module/pytorch3d/pytorch3d/__init__.py)


### Clear and Rebuild the Module
If _C exists but cannot be loaded, the issue might be due to a corrupted build.

Navigate to the PyTorch3D folder:
```bash
cd /home/qian.feng/workspace/SparseDFF/thirdparty_module/pytorch3d
rm -rf build pytorch3d.egg-info
pip install -e .
```
