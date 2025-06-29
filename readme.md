# DeepPotato - GPU

### Demos
#### Available demos (targets)
- **xor_demo**: A simple neural network that learns the XOR function.
- **compression_demo**: A neural network that learns the image.
- **digits_demo**: A neural network that learns the MNIST dataset. (WIP)

#### Building the demos
```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release --target xor_demo
```
