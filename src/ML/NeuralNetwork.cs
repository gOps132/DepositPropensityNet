using System;
using System.Globalization;
using System.IO;

namespace DepositPropensityNet.ML {
	/// <summary>
	/// A 3-layer feed-forward Multi-Layer Perceptron (Input, Hidden, and Output layers)
	/// using sigmoid activations and trained via stochastic gradient descent (backpropagation).
	/// </summary>
	public class NeuralNetwork {
		private readonly InputNeuron[] inputNeurons;
		private readonly HiddenNeuron[] hiddenNeurons;
		private readonly OutputNeuron[] outputNeurons;

		private readonly double outputLearningRate;
		private readonly double inputLearningRate;

		private readonly double[] outputErrors;
		private readonly double[] outputErrorDerivatives;
		private readonly double[] desiredOutputs;

		/// <summary>
		/// Creates the default 47-32-1 binary classification network for Bank Marketing deposit propensity.
		/// </summary>
		public NeuralNetwork()
			: this(47, 32, 1, 0.05, 0.05, 1234) {
		}

		public NeuralNetwork(
			int inputCount,
			int hiddenCount,
			int outputCount,
			double outputLearningRate,
			double inputLearningRate,
			int seed) {
			if (inputCount <= 0) throw new ArgumentOutOfRangeException(nameof(inputCount));
			if (hiddenCount <= 0) throw new ArgumentOutOfRangeException(nameof(hiddenCount));
			if (outputCount <= 0) throw new ArgumentOutOfRangeException(nameof(outputCount));

			this.outputLearningRate = outputLearningRate;
			this.inputLearningRate = inputLearningRate;

			inputNeurons = new InputNeuron[inputCount];
			hiddenNeurons = new HiddenNeuron[hiddenCount];
			outputNeurons = new OutputNeuron[outputCount];
			desiredOutputs = new double[outputCount];
			outputErrors = new double[outputCount];
			outputErrorDerivatives = new double[outputCount];
			CreateNeurons(new Random(seed));
		}

		private void CreateNeurons(Random random) {
			for (int inputIndex = 0; inputIndex < inputNeurons.Length; inputIndex++) {
				inputNeurons[inputIndex] = new InputNeuron(hiddenNeurons.Length, random);
			}
			for (int hiddenIndex = 0; hiddenIndex < hiddenNeurons.Length; hiddenIndex++) {
				hiddenNeurons[hiddenIndex] = new HiddenNeuron(outputNeurons.Length, random);
			}
			for (int outputIndex = 0; outputIndex < outputNeurons.Length; outputIndex++) {
				outputNeurons[outputIndex] = new OutputNeuron(outputIndex);
			}
		}

		public double GetOutput(int outputIndex = 0) {
			return outputNeurons[outputIndex].GetActivation();
		}

		public void SetInput(int inputIndex, double value) {
			inputNeurons[inputIndex].SetInput(value);
		}

		public void SetDesiredOutput(int outputIndex, double value) {
			desiredOutputs[outputIndex] = value;
		}

		/// <summary>
		/// Returns 0.5 * mean((desired - actual)^2) for the current forward pass.
		/// </summary>
		public double CalculateLoss() {
			double sum = 0.0;
			for (int outputIndex = 0; outputIndex < outputNeurons.Length; outputIndex++) {
				double difference = desiredOutputs[outputIndex] - outputNeurons[outputIndex].GetActivation();
				sum += difference * difference;
			}
			return 0.5 * sum / outputNeurons.Length;
		}

		/// <summary>
		/// Applies the logistic sigmoid activation function: σ(z) = 1 / (1 + e^-z).
		/// Clamps extreme values to avoid exponential overflow.
		/// </summary>
		private static double Sigmoid(double value) {
			if (value >= 20.0) {
				return 1.0 / (1.0 + Math.Exp(-32.0));
			}
			if (value <= -20.0) {
				return 1.0 / (1.0 + Math.Exp(32.0));
			}
			return 1.0 / (1.0 + Math.Exp(-value));
		}

		/// <summary>
		/// Calculates the sigmoid activation of each hidden neuron:
		/// a_j = σ(Σ_i (x_i * w_ij) + b_j).
		/// </summary>
		public void CalculateHiddenActivations() {
			for (int hiddenIndex = 0; hiddenIndex < hiddenNeurons.Length; hiddenIndex++) {
				double weightedSum = 0.0;
				for (int inputIndex = 0; inputIndex < inputNeurons.Length; inputIndex++) {
					weightedSum += inputNeurons[inputIndex].GetInput() * inputNeurons[inputIndex].GetWeight(hiddenIndex);
				}
				hiddenNeurons[hiddenIndex].SetActivation(Sigmoid(weightedSum + hiddenNeurons[hiddenIndex].GetBias()));
			}
		}

		/// <summary>
		/// Calculates the sigmoid activation of each output neuron:
		/// y_k = σ(Σ_j (a_j * w_jk) + b_k).
		/// </summary>
		public void CalculateOutputActivations() {
			for (int outputIndex = 0; outputIndex < outputNeurons.Length; outputIndex++) {
				double weightedSum = 0.0;
				for (int hiddenIndex = 0; hiddenIndex < hiddenNeurons.Length; hiddenIndex++) {
					weightedSum += hiddenNeurons[hiddenIndex].GetActivation() * hiddenNeurons[hiddenIndex].GetWeight(outputIndex);
				}
				outputNeurons[outputIndex].SetActivation(Sigmoid(weightedSum + outputNeurons[outputIndex].GetBias()));
			}
		}

		/// <summary>
		/// Calculates the raw output error (desired - actual) for each output neuron.
		/// </summary>
		public void CalculateOutputErrors() {
			for (int outputIndex = 0; outputIndex < outputNeurons.Length; outputIndex++) {
				outputErrors[outputIndex] = desiredOutputs[outputIndex] - outputNeurons[outputIndex].GetActivation();
			}
		}

		/// <summary>
		/// Calculates the output error delta for each output neuron:
		/// δ_k = sampleWeight * y_k * (1 - y_k) * (t_k - y_k).
		/// </summary>
		public void CalculateOutputErrorDerivatives(double sampleWeight = 1.0) {
			for (int outputIndex = 0; outputIndex < outputNeurons.Length; outputIndex++) {
				double activation = outputNeurons[outputIndex].GetActivation();
				outputErrorDerivatives[outputIndex] = sampleWeight * activation * (1.0 - activation) * outputErrors[outputIndex];
			}
		}

		/// <summary>
		/// Runs one stochastic gradient descent training step (forward pass + backpropagation + weight updates)
		/// and returns the unweighted mean squared error for this sample before the weights were updated.
		/// </summary>
		public double TrainStep(double sampleWeight = 1.0) {
			ForwardPass();
			double loss = CalculateLoss();
			CalculateOutputErrors();
			CalculateOutputErrorDerivatives(sampleWeight);

			// 1. Backpropagate error to each hidden neuron.
			for (int hiddenIndex = 0; hiddenIndex < hiddenNeurons.Length; hiddenIndex++) {
				hiddenNeurons[hiddenIndex].CalculateError(outputErrorDerivatives);
			}

			// 2. Update weights between the hidden and output layers.
			for (int hiddenIndex = 0; hiddenIndex < hiddenNeurons.Length; hiddenIndex++) {
				hiddenNeurons[hiddenIndex].UpdateWeights(outputLearningRate, outputErrorDerivatives);
			}

			// 3. Update weights between the input and hidden layers.
			for (int inputIndex = 0; inputIndex < inputNeurons.Length; inputIndex++) {
				for (int hiddenIndex = 0; hiddenIndex < hiddenNeurons.Length; hiddenIndex++) {
					inputNeurons[inputIndex].UpdateWeight(hiddenIndex, hiddenNeurons[hiddenIndex].GetError(), inputLearningRate);
				}
			}

			// 4. Update the output neuron biases.
			for (int outputIndex = 0; outputIndex < outputNeurons.Length; outputIndex++) {
				outputNeurons[outputIndex].UpdateBias(outputLearningRate, outputErrorDerivatives);
			}

			// 5. Update the hidden neuron biases.
			for (int hiddenIndex = 0; hiddenIndex < hiddenNeurons.Length; hiddenIndex++) {
				hiddenNeurons[hiddenIndex].UpdateBias(inputLearningRate);
			}

			return loss;
		}

		/// <summary>
		/// Runs the forward pass through the hidden and output layers.
		/// </summary>
		public void ForwardPass() {
			CalculateHiddenActivations();
			CalculateOutputActivations();
		}

		/// <summary>
		/// Saves all input-to-hidden weights, hidden-to-output weights, hidden biases, and output biases to a file.
		/// </summary>
		public void SaveWeights(string filePath) {
			string? directory = Path.GetDirectoryName(filePath);
			if (!string.IsNullOrEmpty(directory)) {
				Directory.CreateDirectory(directory);
			}

			using (StreamWriter writer = new StreamWriter(filePath)) {
				for (int inputIndex = 0; inputIndex < inputNeurons.Length; inputIndex++) {
					for (int hiddenIndex = 0; hiddenIndex < hiddenNeurons.Length; hiddenIndex++) {
						writer.WriteLine(inputNeurons[inputIndex].GetWeight(hiddenIndex).ToString("R", CultureInfo.InvariantCulture));
					}
				}
				for (int hiddenIndex = 0; hiddenIndex < hiddenNeurons.Length; hiddenIndex++) {
					for (int outputIndex = 0; outputIndex < outputNeurons.Length; outputIndex++) {
						writer.WriteLine(hiddenNeurons[hiddenIndex].GetWeight(outputIndex).ToString("R", CultureInfo.InvariantCulture));
					}
				}
				for (int hiddenIndex = 0; hiddenIndex < hiddenNeurons.Length; hiddenIndex++) {
					writer.WriteLine(hiddenNeurons[hiddenIndex].GetBias().ToString("R", CultureInfo.InvariantCulture));
				}
				for (int outputIndex = 0; outputIndex < outputNeurons.Length; outputIndex++) {
					writer.WriteLine(outputNeurons[outputIndex].GetBias().ToString("R", CultureInfo.InvariantCulture));
				}
			}
		}

		/// <summary>
		/// Loads all weights and biases from a file written by <see cref="SaveWeights"/>.
		/// </summary>
		public void LoadWeights(string filePath) {
			using (StreamReader reader = new StreamReader(filePath)) {
				for (int inputIndex = 0; inputIndex < inputNeurons.Length; inputIndex++) {
					for (int hiddenIndex = 0; hiddenIndex < hiddenNeurons.Length; hiddenIndex++) {
						inputNeurons[inputIndex].SetWeight(hiddenIndex, ParseLine(reader, filePath));
					}
				}
				for (int hiddenIndex = 0; hiddenIndex < hiddenNeurons.Length; hiddenIndex++) {
					for (int outputIndex = 0; outputIndex < outputNeurons.Length; outputIndex++) {
						hiddenNeurons[hiddenIndex].SetWeight(outputIndex, ParseLine(reader, filePath));
					}
				}
				for (int hiddenIndex = 0; hiddenIndex < hiddenNeurons.Length; hiddenIndex++) {
					hiddenNeurons[hiddenIndex].SetBias(ParseLine(reader, filePath));
				}
				for (int outputIndex = 0; outputIndex < outputNeurons.Length; outputIndex++) {
					outputNeurons[outputIndex].SetBias(ParseLine(reader, filePath));
				}
			}
		}

		private static double ParseLine(StreamReader reader, string filePath) {
			string? line = reader.ReadLine();
			if (line == null) {
				throw new InvalidDataException($"Unexpected end of weights file: {filePath}");
			}
			return double.Parse(line, CultureInfo.InvariantCulture);
		}
	}
}
