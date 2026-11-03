using System;
using System.Globalization;
using System.IO;

namespace DepositPropensityNet.ETL {
	/// <summary>
	/// MinMax normalizer that scales continuous numerical features (indices 0..continuousCount-1)
	/// into the [0, 1] range. Fits strictly on the training split to prevent data leakage from
	/// the test/validation split, and clamps out-of-range test values to [0, 1].
	/// </summary>
	public sealed class Normalizer {
		private readonly int continuousCount;
		private readonly double[] minimums;
		private readonly double[] maximums;
		private bool isFitted;

		public int ContinuousCount => continuousCount;
		public bool IsFitted => isFitted;

		public Normalizer(int continuousCount = 6) {
			if (continuousCount <= 0) {
				throw new ArgumentOutOfRangeException(nameof(continuousCount));
			}
			this.continuousCount = continuousCount;
			minimums = new double[continuousCount];
			maximums = new double[continuousCount];
		}

		public double GetMinimum(int featureIndex) => minimums[featureIndex];
		public double GetMaximum(int featureIndex) => maximums[featureIndex];

		/// <summary>
		/// Computes per-feature minimum and maximum values across the training dataset only.
		/// </summary>
		public void Fit(double[][] trainInputs) {
			if (trainInputs == null || trainInputs.Length == 0) {
				throw new ArgumentException("Training inputs cannot be null or empty.", nameof(trainInputs));
			}

			for (int col = 0; col < continuousCount; col++) {
				minimums[col] = double.PositiveInfinity;
				maximums[col] = double.NegativeInfinity;
			}

			for (int row = 0; row < trainInputs.Length; row++) {
				double[] sample = trainInputs[row];
				for (int col = 0; col < continuousCount; col++) {
					double value = sample[col];
					if (value < minimums[col]) minimums[col] = value;
					if (value > maximums[col]) maximums[col] = value;
				}
			}

			isFitted = true;
		}

		/// <summary>
		/// Normalizes a dataset in-place using the previously fitted Min/Max bounds.
		/// Values are clamped to [0, 1] so unseen test samples outside training bounds remain stable.
		/// </summary>
		public void TransformInPlace(double[][] inputs) {
			EnsureFitted();
			if (inputs == null) {
				throw new ArgumentNullException(nameof(inputs));
			}

			for (int row = 0; row < inputs.Length; row++) {
				TransformRowInPlace(inputs[row]);
			}
		}

		/// <summary>
		/// Fits the normalizer on the training set and normalizes it in-place.
		/// </summary>
		public void FitTransformInPlace(double[][] trainInputs) {
			Fit(trainInputs);
			TransformInPlace(trainInputs);
		}

		/// <summary>
		/// Normalizes a single feature vector in-place using fitted Min/Max bounds.
		/// Binary and one-hot features at indices >= continuousCount are already in {0, 1}
		/// and are left untouched.
		/// </summary>
		public void TransformRowInPlace(double[] features) {
			EnsureFitted();
			if (features == null) {
				throw new ArgumentNullException(nameof(features));
			}

			for (int col = 0; col < continuousCount; col++) {
				double range = maximums[col] - minimums[col];
				if (range <= 0.0) {
					features[col] = 0.0;
				}
				else {
					double scaled = (features[col] - minimums[col]) / range;
					features[col] = Math.Clamp(scaled, 0.0, 1.0);
				}
			}
		}

		/// <summary>
		/// Saves the fitted Min/Max parameters to a text file so inference/GUI can apply
		/// the exact same scaling without reloading the training dataset.
		/// </summary>
		public void Save(string filePath) {
			EnsureFitted();
			string? directory = Path.GetDirectoryName(filePath);
			if (!string.IsNullOrEmpty(directory)) {
				Directory.CreateDirectory(directory);
			}

			using (StreamWriter writer = new StreamWriter(filePath)) {
				writer.WriteLine(continuousCount.ToString(CultureInfo.InvariantCulture));
				for (int col = 0; col < continuousCount; col++) {
					writer.WriteLine(FormattableString.Invariant($"{minimums[col]:R},{maximums[col]:R}"));
				}
			}
		}

		/// <summary>
		/// Loads fitted Min/Max parameters from a file written by <see cref="Save"/>.
		/// </summary>
		public static Normalizer Load(string filePath) {
			using (StreamReader reader = new StreamReader(filePath)) {
				string? header = reader.ReadLine();
				if (header == null) {
					throw new InvalidDataException($"Empty normalizer file: {filePath}");
				}

				int count = int.Parse(header, CultureInfo.InvariantCulture);
				Normalizer normalizer = new Normalizer(count);
				for (int col = 0; col < count; col++) {
					string? line = reader.ReadLine();
					if (line == null) {
						throw new InvalidDataException($"Unexpected end of normalizer file: {filePath}");
					}
					string[] parts = line.Split(',');
					normalizer.minimums[col] = double.Parse(parts[0], CultureInfo.InvariantCulture);
					normalizer.maximums[col] = double.Parse(parts[1], CultureInfo.InvariantCulture);
				}

				normalizer.isFitted = true;
				return normalizer;
			}
		}

		private void EnsureFitted() {
			if (!isFitted) {
				throw new InvalidOperationException("Normalizer must be fitted on training data before transforming.");
			}
		}
	}
}
