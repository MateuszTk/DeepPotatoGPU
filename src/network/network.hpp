#pragma once

#include "external.hpp"
#include "layer.hpp"
#include "timer.hpp"

class Network {

	private:

		__host__ __device__ static float sigmoid(float x) {
			return 1.0f / (1.0f + exp(-x));
		}

		__host__ __device__ static float sigmoidDerivative(float x) {
			float s = sigmoid(x);
			return s * (1.0f - s);
		}

		__host__ __device__ static float deriverate(float input, Activation activation) {
			if (activation == Activation::Sigmoid) {
				return sigmoidDerivative(input);
			}
			else {
				return 1.0f;
			}
		}

	private:

		std::vector<Layer> layers;
		uint32_t maxBatchSize;
		uint32_t maxTrainBatchSize;

		float randomNormalizedFloat() {
			float random = ((float)(rand() % RAND_MAX)) / (float)RAND_MAX;
			float nr = random * 2.0f - 1.0f;
			return nr;
		}

		void initRandom(Matrix2D<float>& matrix) {
			for (unsigned int y = 0; y < matrix.shape(0); y++) {
				for (unsigned int x = 0; x < matrix.shape(1); x++) {
					matrix(y, x) = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
				}
			}
		}

	public:

		Network(std::initializer_list<LayerType> layerTypes, uint32_t maxBatchSize, uint32_t maxTrainBatchSize) 
			: maxBatchSize(maxBatchSize), maxTrainBatchSize(maxTrainBatchSize) {
			layers.reserve(layerTypes.size());

			uint32_t inputSize = 0;

			if (maxTrainBatchSize > maxBatchSize) {
				maxBatchSize = maxTrainBatchSize;
			}

			for (const LayerType& layerType : layerTypes) {
				layers.emplace_back(layerType, inputSize, maxBatchSize, maxTrainBatchSize);
				inputSize = layerType.getNeurons();
			}
		}

		virtual ~Network() = default;

		void initialize() {
			/*for (Layer& layer : layers) {
				initRandom(layer.weights);
				initRandom(layer.biases);
			}*/
			int iLayer = 0;
			for (Layer& layer : layers) {
				if (iLayer++ == 0) continue;
				
				for (int i = 0; i < layer.weights.shape(1); i++) {
					layers[iLayer - 2].biases(i) = randomNormalizedFloat();
					for (int j = 0; j < layer.weights.shape(0); j++) {
						layer.weights(j, i) = randomNormalizedFloat();
					}
				}
				//std::cout << layer.weights;
			}
			for (int i = 0; i < layers.back().biases.shape(0); i++) {
				layers.back().biases(i) = randomNormalizedFloat();
			}
		}

		/*
		* Forward
		*/

		GENERIC_KERNEL(ForwardFirstLayerKernel) {

			__host__ __device__ float activate(float input, Activation activation) {
				if (activation == Activation::Sigmoid) {
					return sigmoid(input);
				}
				else {
					return input;
				}
			}

			GENERIC_KERNEL_ENTRY(Layer::InputsMat_t inputs, Matrix2D<float> currentInputs, Layer::OutputsMat_t outputs, Activation activation, uint32_t offset, uint32_t batchSize) {
				uint3 index = getThreadIdx() + getBlockIdx() * getBlockDim();

				if (index.x >= 1 || index.y >= outputs.shape(1) || index.z >= batchSize) {
					return;
				}

				// If is first layer, the weights are empty and the input should be forwarded directly to the output
				Layer::OutputsMat_t::type output = currentInputs(index.z + offset, index.y);

				inputs(index.z, index.y) = output;
				outputs(index.z, index.y) = activate(output, activation);
			}
		};

		GENERIC_KERNEL(ForwardLayerKernel) {

			__host__ __device__ float activate(float input, Activation activation) {
				if (activation == Activation::Sigmoid) {
					return sigmoid(input);
				}
				else {
					return input;
				}
			}

			GENERIC_KERNEL_ENTRY(Layer::WeightsMat_t weights, Layer::BiasesMat_t biases, Layer::InputsMat_t inputs, Layer::OutputsMat_t currentInputs, Layer::OutputsMat_t outputs, Activation activation, uint32_t offset, uint32_t batchSize) {
				#if defined(__CUDA_ARCH__) && USE_WMMA == 1
				int tileM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
				int tileN = blockIdx.y;

				int M = weights.dataShape(0);
				int N = currentInputs.dataShape(0);
				int K = weights.dataShape(1);

				wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
				wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
				wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
				//wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> out_frag;

				wmma::fill_fragment(c_frag, 0.0f);

				for (int tileK = 0; tileK < K / WMMA_K; tileK++) {
					const half* tile_a = &weights(tileM * WMMA_M, tileK * WMMA_K);
					const half* tile_b = &currentInputs(tileN * WMMA_N, tileK * WMMA_K);

					wmma::load_matrix_sync(a_frag, tile_a, K);
					wmma::load_matrix_sync(b_frag, tile_b, K);

					wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
				}

				// Store to shared memory
				//__shared__ float inputs_shared[WMMA_M * WMMA_N];
				//wmma::store_matrix_sync(inputs_shared, c_frag, WMMA_N, wmma::mem_row_major);

				float* c_tile = &inputs(tileN * WMMA_N, tileM * WMMA_M);
				wmma::store_matrix_sync(c_tile, c_frag, M, wmma::mem_col_major);
				
				float bias = biases(tileM * WMMA_M + threadIdx.x / 2);
				for (int i = 0; i < c_frag.num_elements; i++) {
					int xp = tileN * WMMA_N + i + (threadIdx.x % 2) * c_frag.num_elements;
					int yp = tileM * WMMA_M + threadIdx.x / 2;
					auto& input = inputs(xp, yp);
					//float input = inputs_shared[xp + yp * WMMA_N] + bias;
					input += bias;
					//inputs(xp, yp) = input;
					outputs(xp, yp) = __float2half(activate(input, activation));
				}

				/*// maybe todo merge weights and biases into one matrix
				float* c_tile = &inputs(tileN * WMMA_N, tileM * WMMA_M);
				for (int i = 0; i < c_frag.num_elements; i++) {
					c_frag.x[i] += biases(tileM * WMMA_M);
				}
				wmma::store_matrix_sync(c_tile, c_frag, K, wmma::mem_col_major);

				for (int i = 0; i < out_frag.num_elements; i++) {
					out_frag.x[i] = __float2half(activate(c_frag.x[i], activation));
				}

				lowp_t* output_tile = &outputs(tileN * WMMA_N, tileM * WMMA_M);
				wmma::store_matrix_sync(output_tile, out_frag, K, wmma::mem_col_major);*/

				#else

				uint3 index = getThreadIdx() + getBlockIdx() * getBlockDim();

				if (index.x >= 1 || index.y >= outputs.shape(1) || index.z >= batchSize) {
					return;
				}

				float output = 0.0f;

				for (unsigned int i = 0; i < weights.shape(1); i++) {
					output += static_cast<float>(weights(index.y, i) * currentInputs(index.z, i));
				}

				output += biases(index.y);

				inputs(index.z, index.y) = output;
				outputs(index.z, index.y) = activate(output, activation);
				#endif
			}
		};

		template <typename... Args>
		void runForwardLayerKernel(CUDAExecutor& executor, dim3 size, Args&... args) {
			#if USE_WMMA == 1
			auto launchInfo = Matrix2D<float>::getWMMALaunchSize(size.y, size.z);
			executor.executeParams<ForwardLayerKernel>(launchInfo, args...);
			#else
			executor.execute<ForwardLayerKernel>(size, args...);
			#endif
		}

		template <typename... Args>
		void runForwardLayerKernel(CPUExecutor& executor, dim3 size, Args&... args) {
			executor.execute<ForwardLayerKernel>(size, args...);
		}

		template <typename Exe>
		void forward(Exe& executor, Matrix2D<float>& input, uint32_t batchSize = -1, uint32_t offset = 0) {

			if (batchSize == -1) {
				batchSize = input.shape(0);
			}

			if (batchSize > layers[0].outputs.shape(0)) {
				throw std::invalid_argument("Input batch size must be no greater than the specified maximum network batch size");
			}

			if (input.shape(0) < batchSize) {
				throw std::invalid_argument("Input batch size must be no less than the specified maximum network batch size");
			}			

			Activation activation = layers[0].type.getActivation();
			executor.template execute<ForwardFirstLayerKernel>({ 1, layers[0].outputs.shape(1), batchSize },
				layers[0].inputs, input, layers[0].outputs, activation, offset, batchSize
			);
			
			Layer::OutputsMat_t currentInput = layers[0].outputs;
			
			for (int i = 1; i < layers.size(); i++) {
				Layer& layer = layers[i];

				Activation activation = layer.type.getActivation();
				runForwardLayerKernel(executor, { 1, layer.outputs.shape(1), batchSize },
					layer.weights, layer.biases, layer.inputs, currentInput, layer.outputs, activation, offset, batchSize
				);
				
				currentInput = layer.outputs;
			}	
		}

		/**
		* Backward
		*/

		GENERIC_KERNEL(OutputLayerErrorKernel) {
			GENERIC_KERNEL_ENTRY(Matrix2D<float> target, Layer::OutputsMat_t output, Layer::InputsMat_t input, Layer::ErrorsMat_t error, Activation activation, uint32_t offset, uint32_t batchSize) {
				uint3 index = getThreadIdx() + getBlockIdx() * getBlockDim();

				if (index.x >= 1 || index.y >= error.shape(1) || index.z >= batchSize) {
					return;
				}

				error(index.z, index.y, 0) = (target(index.z + offset, index.y) - static_cast<float>(output(index.z, index.y))) * deriverate(input(index.z, index.y), activation);
			}
		};

		GENERIC_KERNEL(BackwardLayerKernel) {
			GENERIC_KERNEL_ENTRY(Layer::WeightsMat_t weights, Layer::ErrorsMat_t errors, Layer::ErrorsMat_t prevErrors, Layer::InputsMat_t prevOutputs, Activation activation, uint32_t offset, uint32_t batchSize) {
				uint3 index = getThreadIdx() + getBlockIdx() * getBlockDim();

				if (index.x >= 1 || index.y >= prevErrors.shape(1) || index.z >= batchSize) {
					return;
				}

				float sum = 0.0f;

				for (unsigned int i = 0; i < weights.shape(0); i++) {
					sum += static_cast<float>(weights(i, index.y)) * errors(index.z, i, 0);
				}

				sum *= deriverate(prevOutputs(index.z, index.y), activation);

				prevErrors(index.z, index.y, 0) = sum;
			}
		};

		template <typename Exe>
		void backward(Exe& executor, Matrix2D<float>& target, uint32_t batchSize = -1, uint32_t offset = 0) {

			if (batchSize == -1) {
				batchSize = target.shape(0);
			}

			if (target.shape(0) < batchSize) {
				throw std::invalid_argument("Target batch size must be no less than the specified maximum network batch size");
			}

			Layer& outputLayer = layers.back();

			Activation activation = outputLayer.type.getActivation();
			executor.template execute<OutputLayerErrorKernel>({ 1, outputLayer.errors.shape(1), batchSize },
				target, outputLayer.outputs, outputLayer.inputs, outputLayer.errors, activation, offset, batchSize
			);

			for (int i = layers.size() - 2; i > 0; i--) {
				Layer& layer = layers[i];
				Layer& nextLayer = layers[i + 1];

				Activation activation = layer.type.getActivation();
				executor.template execute<BackwardLayerKernel>({ 1, layer.errors.shape(1), batchSize },
					nextLayer.weights, nextLayer.errors, layer.errors, layer.inputs, activation, offset, batchSize
				);
			}
		}

		/**
		* Update weights and biases
		*/

		GENERIC_KERNEL(UpdateWeightsAndBiasesKernel) {
			GENERIC_KERNEL_ENTRY(Layer::WeightsMat_t weights, Layer::BiasesMat_t biases, Layer::ErrorsMat_t errors, Layer::OutputsMat_t prevOutputs, float learningRate, unsigned int updateBatchSize, uint32_t offset) {
				uint3 index = getThreadIdx() + getBlockIdx() * getBlockDim();

				if (index.x >= prevOutputs.shape(1) || index.y >= weights.shape(0)) {
					return;
				}

				for (unsigned int batch = 0; batch < updateBatchSize; batch++) {
					float error = learningRate * errors(batch, index.y, 0);
					weights(index.y, index.x) += error * static_cast<float>(prevOutputs(batch, index.x));
					if (index.x == 0) {
						biases(index.y) += error;
					}
				}
			}
		};

		/*
		GENERIC_KERNEL(UpdateWeightsAndBiasesKernel) {
			GENERIC_KERNEL_ENTRY(Layer::WeightsMat_t weights, Layer::BiasesMat_t biases, Layer::ErrorsMat_t errors, Layer::OutputsMat_t prevOutputs, float learningRate, unsigned int updateBatchSize, uint32_t offset) {
				uint3 index = getThreadIdx() + getBlockIdx() * getBlockDim();

				if (index.x >= 1 || index.y >= weights.shape(0)) {
					return;
				}

				for (unsigned int batch = 0; batch < updateBatchSize; batch++) {
					float error = learningRate * errors(batch, index.y, 0);
					for (unsigned int x = 0; x < prevOutputs.shape(1); x++) {
						weights(index.y, x) += error * static_cast<float>(prevOutputs(batch, x));
					}
						//if (index.x == 0) {
							biases(index.y) += error;
						//}
				}
			}
		};
		*/

		template <typename Exe>
		void update(Exe& executor, float learningRate, uint32_t batchSize, uint32_t offset = 0) {
			if (batchSize > maxBatchSize) {
				throw std::invalid_argument("Batch size must be no greater than the specified maximum network batch size");
			}

			for (int i = 1; i < layers.size(); i++) {
				Layer& layer = layers[i];
				Layer& previousLayer = layers[i - 1];
				//executor.template execute<UpdateWeightsAndBiasesKernel>({ 1, layer.weights.shape(0) },
				//	layer.weights, layer.biases, layer.errors, previousLayer.outputs, learningRate, batchSize, offset
				//);
				executor.template execute<UpdateWeightsAndBiasesKernel>({ previousLayer.outputs.shape(1), layer.weights.shape(0) },
					layer.weights, layer.biases, layer.errors, previousLayer.outputs, learningRate, batchSize, offset
				);
			}
		}

		template <typename Exe>
		void update(Exe& executor, float learningRate, Matrix2D<float>& target) {
			update(executor, learningRate, target.shape(0), 0);
		}

		/**
		* Other
		*/

		Layer::OutputsMat_t& getOutput() {
			return layers.back().outputs;
		}

		uint32_t getMaximumBatchSize() const {
			return maxBatchSize;
		}

		uint32_t getMaximumTrainBatchSize() const {
			return maxTrainBatchSize;
		}
};
