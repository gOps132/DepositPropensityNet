using System;

namespace DepositPropensityNet.ML {
	/// <summary>
	/// A neuron in the input layer. It stores one normalized feature value and
	/// the outgoing weights that connect it to each neuron in the hidden layer.
	/// </summary>
	public class InputNeuron {
		private readonly double[] weights;
		private readonly Random random;
		private double input;

		public InputNeuron(int weightCount, Random random) {
			input = 0.0;
			weights = new double[weightCount];
			this.random = random;
			RandomizeWeights();
		}

		private double RandomWeight() {
			return random.Next(-1000, 1000) / 2000.0;
		}

		private void RandomizeWeights() {
			for (int weightIndex = 0; weightIndex < weights.Length; weightIndex++) {
				weights[weightIndex] = RandomWeight();
			}
		}

		/// <summary>
		/// Updates the weight connecting this input neuron to a specific hidden neuron
		/// using stochastic gradient descent: Δw = η_hidden * δ_hidden * input.
		/// </summary>
		public void UpdateWeight(int hiddenIndex, double error, double learningRate) {
			double scaledError = error * learningRate;
			weights[hiddenIndex] += scaledError * input;
		}

		public double GetInput() {
			return input;
		}

		public void SetInput(double value) {
			input = value;
		}

		public double GetWeight(int hiddenIndex) {
			return weights[hiddenIndex];
		}

		public void SetWeight(int weightIndex, double value) {
			weights[weightIndex] = value;
		}
	}
}
