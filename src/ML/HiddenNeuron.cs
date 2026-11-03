using System;

namespace DepositPropensityNet.ML {
	/// <summary>
	/// A neuron in the hidden layer. It stores a sigmoid activation value, a bias,
	/// a backpropagated error delta, and the outgoing weights connecting it to each output neuron.
	/// </summary>
	public class HiddenNeuron {
		private readonly double[] weights;
		private readonly Random random;
		private double activation;
		private double bias;
		private double error;

		public HiddenNeuron(int weightCount, Random random) {
			activation = 0.0;
			bias = 0.01;
			error = 0.0;
			weights = new double[weightCount];
			this.random = random;
			RandomizeWeights();
		}

		/// <summary>
		/// Calculates this hidden neuron's backpropagated error delta from the output layer:
		/// δ_j = (Σ_k δ_k * w_jk) * a_j * (1 - a_j).
		/// </summary>
		public void CalculateError(double[] outputErrorDerivatives) {
			double weightedSum = 0.0;
			for (int outputIndex = 0; outputIndex < outputErrorDerivatives.Length; outputIndex++) {
				weightedSum += outputErrorDerivatives[outputIndex] * weights[outputIndex];
			}
			error = weightedSum * activation * (1.0 - activation);
		}

		/// <summary>
		/// Updates the outgoing weights between this hidden neuron and the output layer:
		/// Δw_jk = η_output * δ_k * a_j.
		/// </summary>
		public void UpdateWeights(double learningRate, double[] outputErrorDerivatives) {
			double scaledActivation = activation * learningRate;
			for (int outputIndex = 0; outputIndex < outputErrorDerivatives.Length; outputIndex++) {
				weights[outputIndex] += scaledActivation * outputErrorDerivatives[outputIndex];
			}
		}

		public void SetWeight(int outputIndex, double value) {
			weights[outputIndex] = value;
		}

		public double GetWeight(int outputIndex) {
			return weights[outputIndex];
		}

		public double GetError() {
			return error;
		}

		public void SetActivation(double value) {
			activation = value;
		}

		public double GetActivation() {
			return activation;
		}

		public void SetBias(double value) {
			bias = value;
		}

		public double GetBias() {
			return bias;
		}

		/// <summary>
		/// Updates this hidden neuron's bias using its error delta: Δb_j = η_hidden * δ_j.
		/// </summary>
		public void UpdateBias(double learningRate) {
			bias += error * learningRate;
		}

		private double RandomWeight() {
			return random.Next(-1000, 1000) / 2000.0;
		}

		private void RandomizeWeights() {
			for (int weightIndex = 0; weightIndex < weights.Length; weightIndex++) {
				weights[weightIndex] = RandomWeight();
			}
		}
	}
}
