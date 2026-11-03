using System;
using System.Globalization;
using System.Text;

namespace DepositPropensityNet.ML {
	/// <summary>
	/// Binary classification evaluation results for DepositPropensityNet:
	/// overall Accuracy, positive-class ("yes") Precision/Recall/F1, per-class metrics
	/// ("no" = 0, "yes" = 1), macro averages, and a 2x2 confusion matrix indexed as [actual, predicted].
	/// </summary>
	public sealed class ClassificationMetrics {
		public static readonly string[] ClassNames = { "no (0)", "yes (1)" };
		public static readonly string[] ShortClassNames = { "no", "yes" };

		private readonly int[,] confusion;
		private readonly double[] precision;
		private readonly double[] recall;
		private readonly double[] f1;
		private readonly int[] support;

		public int ClassCount => 2;
		public double Accuracy { get; }
		public double MacroPrecision { get; }
		public double MacroRecall { get; }
		public double MacroF1 { get; }

		/// <summary>Precision for the positive class ("yes" = 1): TP / (TP + FP).</summary>
		public double PositivePrecision => precision[1];

		/// <summary>Recall (Sensitivity) for the positive class ("yes" = 1): TP / (TP + FN).</summary>
		public double PositiveRecall => recall[1];

		/// <summary>F1 score for the positive class ("yes" = 1).</summary>
		public double PositiveF1 => f1[1];

		public int TrueNegatives => confusion[0, 0];
		public int FalsePositives => confusion[0, 1];
		public int FalseNegatives => confusion[1, 0];
		public int TruePositives => confusion[1, 1];

		private ClassificationMetrics(
			int[,] confusion,
			double[] precision,
			double[] recall,
			double[] f1,
			int[] support,
			double accuracy,
			double macroPrecision,
			double macroRecall,
			double macroF1) {
			this.confusion = confusion;
			this.precision = precision;
			this.recall = recall;
			this.f1 = f1;
			this.support = support;
			Accuracy = accuracy;
			MacroPrecision = macroPrecision;
			MacroRecall = macroRecall;
			MacroF1 = macroF1;
		}

		public int[,] GetConfusionMatrix() {
			return confusion;
		}

		public static ClassificationMetrics Create(int[] actual, int[] predicted) {
			if (actual == null) throw new ArgumentNullException(nameof(actual));
			if (predicted == null) throw new ArgumentNullException(nameof(predicted));
			if (actual.Length != predicted.Length) {
				throw new ArgumentException("Actual and predicted label counts must match.");
			}

			const int classCount = 2;
			int[,] confusion = new int[classCount, classCount];
			for (int index = 0; index < actual.Length; index++) {
				confusion[actual[index], predicted[index]]++;
			}

			double[] precision = new double[classCount];
			double[] recall = new double[classCount];
			double[] f1 = new double[classCount];
			int[] support = new int[classCount];
			int correct = 0;

			for (int classIndex = 0; classIndex < classCount; classIndex++) {
				int truePos = confusion[classIndex, classIndex];
				int actualCount = 0;
				int predictedCount = 0;

				for (int other = 0; other < classCount; other++) {
					actualCount += confusion[classIndex, other];
					predictedCount += confusion[other, classIndex];
				}

				support[classIndex] = actualCount;
				correct += truePos;
				precision[classIndex] = predictedCount == 0 ? 0.0 : (double)truePos / predictedCount;
				recall[classIndex] = actualCount == 0 ? 0.0 : (double)truePos / actualCount;

				double denominator = precision[classIndex] + recall[classIndex];
				f1[classIndex] = denominator == 0.0
					? 0.0
					: 2.0 * precision[classIndex] * recall[classIndex] / denominator;
			}

			double macroPrecision = (precision[0] + precision[1]) / 2.0;
			double macroRecall = (recall[0] + recall[1]) / 2.0;
			double macroF1 = (f1[0] + f1[1]) / 2.0;
			double accuracy = actual.Length == 0 ? 0.0 : (double)correct / actual.Length;

			return new ClassificationMetrics(
				confusion,
				precision,
				recall,
				f1,
				support,
				accuracy,
				macroPrecision,
				macroRecall,
				macroF1);
		}

		/// <summary>Formats the per-class and binary classification metrics as a console table.</summary>
		public string FormatTable() {
			StringBuilder builder = new StringBuilder();
			builder.AppendLine("class     precision  recall      f1  support");

			for (int classIndex = 0; classIndex < ClassCount; classIndex++) {
				builder.AppendLine(FormattableString.Invariant(
					$"{ClassNames[classIndex],-8}  {precision[classIndex],9:F3}  {recall[classIndex],6:F3}  {f1[classIndex],6:F3}  {support[classIndex],7}"));
			}

			builder.AppendLine(FormattableString.Invariant(
				$"macro     {MacroPrecision,9:F3}  {MacroRecall,6:F3}  {MacroF1,6:F3}  {TotalSupport,7}"));
			builder.AppendLine(FormattableString.Invariant(
				$"confusion: TN={TrueNegatives}  FP={FalsePositives}  FN={FalseNegatives}  TP={TruePositives}"));
			builder.AppendLine(FormattableString.Invariant(
				$"accuracy {Accuracy * 100.0:F2}%  |  yes precision {PositivePrecision * 100.0:F2}%  |  yes recall {PositiveRecall * 100.0:F2}%  |  yes F1 {PositiveF1:F3}"));
			return builder.ToString();
		}

		public string ToMetricsCsv() {
			StringBuilder builder = new StringBuilder();
			builder.AppendLine("class,precision,recall,f1,support");

			for (int classIndex = 0; classIndex < ClassCount; classIndex++) {
				builder.AppendLine(FormattableString.Invariant(
					$"{ShortClassNames[classIndex]},{precision[classIndex]:F6},{recall[classIndex]:F6},{f1[classIndex]:F6},{support[classIndex]}"));
			}

			builder.AppendLine(FormattableString.Invariant(
				$"macro,{MacroPrecision:F6},{MacroRecall:F6},{MacroF1:F6},{TotalSupport}"));
			builder.AppendLine(FormattableString.Invariant(
				$"accuracy,{Accuracy:F6},{Accuracy:F6},{Accuracy:F6},{TotalSupport}"));
			return builder.ToString();
		}

		public string ToConfusionCsv() {
			StringBuilder builder = new StringBuilder();
			builder.Append("actual\\predicted");
			for (int classIndex = 0; classIndex < ClassCount; classIndex++) {
				builder.Append(',');
				builder.Append(ShortClassNames[classIndex]);
			}
			builder.AppendLine();

			for (int actual = 0; actual < ClassCount; actual++) {
				builder.Append(ShortClassNames[actual]);
				for (int predicted = 0; predicted < ClassCount; predicted++) {
					builder.Append(',');
					builder.Append(confusion[actual, predicted].ToString(CultureInfo.InvariantCulture));
				}
				builder.AppendLine();
			}
			return builder.ToString();
		}

		private int TotalSupport => support[0] + support[1];
	}
}
