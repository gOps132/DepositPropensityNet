using System;
using System.Globalization;
using DepositPropensityNet.ETL;

namespace DepositPropensityNet {
	internal static class Program {
		private const string DefaultDatasetPath = "data/bank/bank-full.csv";
		private const double DefaultTrainRatio = 0.80;
		private const int DefaultSeed = 1234;

		private static void Main(string[] args) {
			string datasetPath = args.Length > 0 ? args[0] : DefaultDatasetPath;

			Console.WriteLine($"Loading and encoding '{datasetPath}'...");
			(double[][] allInputs, int[] allLabels) = Loader.LoadCsv(datasetPath);

			int totalPositive = CountPositive(allLabels);
			Console.WriteLine(FormattableString.Invariant(
				$"Loaded {allInputs.Length} rows x {Loader.FeatureCount} features (dropped 'duration')."));
			Console.WriteLine(FormattableString.Invariant(
				$"Target distribution: yes={totalPositive} ({100.0 * totalPositive / allInputs.Length:F2}%), no={allInputs.Length - totalPositive} ({100.0 * (allInputs.Length - totalPositive) / allInputs.Length:F2}%)"));

			Loader.DatasetSplit split = Loader.SplitTrainTest(allInputs, allLabels, DefaultTrainRatio, DefaultSeed);
			int trainPos = CountPositive(split.TrainLabels);
			int testPos = CountPositive(split.TestLabels);

			Console.WriteLine(FormattableString.Invariant(
				$"Split (80/20, seed={DefaultSeed}): train={split.TrainInputs.Length} (yes={100.0 * trainPos / split.TrainInputs.Length:F2}%), test={split.TestInputs.Length} (yes={100.0 * testPos / split.TestInputs.Length:F2}%)"));

			Normalizer normalizer = new Normalizer(Loader.ContinuousFeatureCount);
			normalizer.FitTransformInPlace(split.TrainInputs);
			normalizer.TransformInPlace(split.TestInputs);

			Console.WriteLine("\nFitted MinMax Normalizer on Training Split (continuous features 0..5):");
			for (int i = 0; i < Loader.ContinuousFeatureCount; i++) {
				Console.WriteLine(FormattableString.Invariant(
					$"  [{i}] {Loader.ContinuousFeatureNames[i],-10} min={normalizer.GetMinimum(i),8:F1}  max={normalizer.GetMaximum(i),10:F1}"));
			}

			Console.WriteLine("\nSample normalized training vector [0] (first 9 features + active one-hot indices):");
			PrintSampleSummary(split.TrainInputs[0], split.TrainLabels[0]);
		}

		private static int CountPositive(int[] labels) {
			int count = 0;
			for (int i = 0; i < labels.Length; i++) {
				if (labels[i] == 1) count++;
			}
			return count;
		}

		private static void PrintSampleSummary(double[] features, int label) {
			Console.Write("  continuous/binary [0..8]: ");
			for (int i = 0; i < 9; i++) {
				Console.Write(FormattableString.Invariant($"{features[i]:F4} "));
			}
			Console.WriteLine();

			Console.Write("  active one-hot indices [9..46]: ");
			for (int i = 9; i < features.Length; i++) {
				if (features[i] > 0.5) {
					Console.Write($"{i} ");
				}
			}
			Console.WriteLine($" -> label={label} ({(label == 1 ? "yes" : "no")})");
		}
	}
}
