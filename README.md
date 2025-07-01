# Tiny Torch Mirror

A minimal PyTorch mirror for private remote servers with no network access.

## Installation

```bash
pip install tiny-torch-mirror
```

## Usage

### Sync to Private Repository from PyTorch Index

1. **Create a config file**
   ```bash
   tiny-torch-mirror config
   ```

2. **Edit the config file**  
   Set the `mirror_root` to your private mirror root directory.

3. **Install this package on your private server.**
   ```bash
   # on your private server
   pip install tiny-torch-mirror
   ```

4. **Run the sync command locally.**
   ```bash
   tiny-torch-mirror sync
   ```

### Use the Private Mirror

**Start the mirror server:**

```bash
tiny-torch-mirror serve --path <path to your private mirror> --port 8081
```

**Install packages from the mirror:**  

To install torch and torchvision with CUDA 11.8:

```bash
pip install torch torchvision --index-url http://localhost:8081/whl/cu118
```