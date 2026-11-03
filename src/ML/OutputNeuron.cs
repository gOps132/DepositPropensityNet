namespace DepositPropensityNet.ML {
	/// <summary>
	/// A neuron in the output layer. It stores a sigmoid activation value, a bias,
	/// and an identifier used to look up its output error delta during backpropagation.
	/// </summary>
	public class OutputNeuron {
		private readonly int id;
		private double bias;
		private double activation;

		public OutputNeuron(int id) {
			this.id = id;
			bias = 0.01;
			activation = 0.0;
		}

		public void SetActivation(double value) {
			activation = value;
		}

		public double GetActivation() {
			return activation;
		}

		public double GetBias() {
			return bias;
		}

		public void SetBias(double value) {
			bias = value;
		}

		/// <summary>
		/// Updates this output neuron's bias using its output error delta: Δb_k = η_output * δ_k.
		/// </summary>
		public void UpdateBias(double learningRate, double[] outputErrorDerivatives) {
			bias += learningRate * outputErrorDerivatives[id];
		}
	}
}
