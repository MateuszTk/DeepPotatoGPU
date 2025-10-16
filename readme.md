# DeepPotato - GPU

### Demos
#### Available demos (targets)
- **xor_demo**: A simple neural network that learns the XOR function.
- **compression_demo**: A neural network that learns the image.
- **digits_demo**: A neural network that learns the MNIST dataset.

#### Building the demos
```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release --target xor_demo
```

Available options for CMake:
- `-DDISABLE_CUDA=ON`: Disable CUDA support even if available. (default: OFF)
- `-DENABLE_FP16=ON`: Use FP16 data type. (default: OFF)
- `-DENABLE_WMMA=ON`: Use Tensor Cores via WMMA API. (default: OFF)
