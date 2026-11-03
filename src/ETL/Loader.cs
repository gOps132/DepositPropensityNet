using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;

namespace DepositPropensityNet.ETL {
	/// <summary>
	/// Loads, cleans, encodes, and splits the UCI Bank Marketing dataset (bank-full.csv / bank.csv).
	/// Drops the post-call 'duration' column to prevent data leakage and transforms the remaining
	/// 15 attributes into a 47-feature numerical vector ready for normalization and training.
	/// </summary>
	public static class Loader {
		/// <summary>Total number of encoded input features after dropping 'duration'.</summary>
		public const int FeatureCount = 47;

		/// <summary>Number of continuous numerical features placed at indices 0..5.</summary>
		public const int ContinuousFeatureCount = 6;

		// Expected 17 columns in the raw UCI Bank Marketing CSV.
		private const int RawColumnCount = 17;

		// Raw column indices in bank-full.csv:
		// 0:age, 1:job, 2:marital, 3:education, 4:default, 5:balance, 6:housing, 7:loan,
		// 8:contact, 9:day, 10:month, 11:duration (DROPPED), 12:campaign, 13:pdays,
		// 14:previous, 15:poutcome, 16:y
		private const int ColAge = 0;
		private const int ColJob = 1;
		private const int ColMarital = 2;
		private const int ColEducation = 3;
		private const int ColDefault = 4;
		private const int ColBalance = 5;
		private const int ColHousing = 6;
		private const int ColLoan = 7;
		private const int ColContact = 8;
		private const int ColDay = 9;
		private const int ColMonth = 10;
		private const int ColDuration = 11; // Intentionally excluded to prevent data leakage.
		private const int ColCampaign = 12;
		private const int ColPdays = 13;
		private const int ColPrevious = 14;
		private const int ColPoutcome = 15;
		private const int ColTarget = 16;

		public static readonly string[] ContinuousFeatureNames = {
			"age", "balance", "day", "campaign", "pdays", "previous"
		};

		public static readonly string[] JobCategories = {
			"admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired",
			"self-employed", "services", "student", "technician", "unemployed", "unknown"
		};

		public static readonly string[] MaritalCategories = {
			"divorced", "married", "single"
		};

		public static readonly string[] EducationCategories = {
			"primary", "secondary", "tertiary", "unknown"
		};

		public static readonly string[] ContactCategories = {
			"cellular", "telephone", "unknown"
		};

		public static readonly string[] MonthCategories = {
			"jan", "feb", "mar", "apr", "may", "jun",
			"jul", "aug", "sep", "oct", "nov", "dec"
		};

		public static readonly string[] PoutcomeCategories = {
			"failure", "other", "success", "unknown"
		};

		/// <summary>
		/// Holds the split dataset before or after normalization.
		/// </summary>
		public sealed class DatasetSplit {
			public double[][] TrainInputs { get; }
			public int[] TrainLabels { get; }
			public double[][] TestInputs { get; }
			public int[] TestLabels { get; }

			public DatasetSplit(double[][] trainInputs, int[] trainLabels, double[][] testInputs, int[] testLabels) {
				TrainInputs = trainInputs;
				TrainLabels = trainLabels;
				TestInputs = testInputs;
				TestLabels = testLabels;
			}
		}

		/// <summary>
		/// Reads a semicolon-delimited Bank Marketing CSV file, skips the header row,
		/// drops the 'duration' column, and encodes features and binary labels.
		/// </summary>
		public static (double[][] inputs, int[] labels) LoadCsv(string csvPath) {
			if (!File.Exists(csvPath)) {
				throw new FileNotFoundException($"Dataset file not found: {csvPath}", csvPath);
			}

			List<double[]> inputsList = new List<double[]>(46000);
			List<int> labelsList = new List<int>(46000);

			using (StreamReader reader = new StreamReader(csvPath)) {
				string? headerLine = reader.ReadLine();
				if (headerLine == null) {
					throw new InvalidDataException($"Dataset file is empty: {csvPath}");
				}

				int lineNumber = 1;
				string? line;
				while ((line = reader.ReadLine()) != null) {
					lineNumber++;
					if (string.IsNullOrWhiteSpace(line)) {
						continue;
					}

					string[] tokens = SplitCsvLine(line);
					if (tokens.Length != RawColumnCount) {
						throw new InvalidDataException(
							$"Line {lineNumber} in '{csvPath}' has {tokens.Length} columns; expected {RawColumnCount}.");
					}

					double[] features = EncodeRow(tokens, lineNumber);
					int label = ParseBinary(tokens[ColTarget], "y", lineNumber);

					inputsList.Add(features);
					labelsList.Add(label);
				}
			}

			return (inputsList.ToArray(), labelsList.ToArray());
		}

		/// <summary>
		/// Encodes a single raw 17-column token array into the 47-element feature vector
		/// (excluding column 11 'duration').
		/// Layout:
		///   [0..5]   : 6 continuous features (age, balance, day, campaign, pdays, previous)
		///   [6..8]   : 3 binary features (default, housing, loan)
		///   [9..20]  : 12 one-hot features for job
		///   [21..23] : 3 one-hot features for marital
		///   [24..27] : 4 one-hot features for education
		///   [28..30] : 3 one-hot features for contact
		///   [31..42] : 12 one-hot features for month
		///   [43..46] : 4 one-hot features for poutcome
		/// </summary>
		public static double[] EncodeRow(string[] tokens, int lineNumber = 0) {
			double[] features = new double[FeatureCount];
			int offset = 0;

			// 1. Continuous numerical features (indices 0..5)
			features[offset++] = ParseDouble(tokens[ColAge], "age", lineNumber);
			features[offset++] = ParseDouble(tokens[ColBalance], "balance", lineNumber);
			features[offset++] = ParseDouble(tokens[ColDay], "day", lineNumber);
			features[offset++] = ParseDouble(tokens[ColCampaign], "campaign", lineNumber);
			features[offset++] = ParseDouble(tokens[ColPdays], "pdays", lineNumber);
			features[offset++] = ParseDouble(tokens[ColPrevious], "previous", lineNumber);

			// 2. Binary features ("yes" -> 1.0, "no" -> 0.0) (indices 6..8)
			features[offset++] = ParseBinary(tokens[ColDefault], "default", lineNumber);
			features[offset++] = ParseBinary(tokens[ColHousing], "housing", lineNumber);
			features[offset++] = ParseBinary(tokens[ColLoan], "loan", lineNumber);

			// 3. Multi-category one-hot features (indices 9..46)
			WriteOneHot(features, ref offset, tokens[ColJob], JobCategories, "job", lineNumber);
			WriteOneHot(features, ref offset, tokens[ColMarital], MaritalCategories, "marital", lineNumber);
			WriteOneHot(features, ref offset, tokens[ColEducation], EducationCategories, "education", lineNumber);
			WriteOneHot(features, ref offset, tokens[ColContact], ContactCategories, "contact", lineNumber);
			WriteOneHot(features, ref offset, tokens[ColMonth], MonthCategories, "month", lineNumber);
			WriteOneHot(features, ref offset, tokens[ColPoutcome], PoutcomeCategories, "poutcome", lineNumber);

			return features;
		}

		/// <summary>
		/// Randomly shuffles and splits the dataset into a Training Set (e.g., 80%) and
		/// Testing/Validation Set (e.g., 20%) using Fisher-Yates shuffle with a fixed seed.
		/// Clones each feature array so in-place normalization does not mutate shared references.
		/// </summary>
		public static DatasetSplit SplitTrainTest(
			double[][] allInputs,
			int[] allLabels,
			double trainRatio,
			int seed) {
			if (allInputs == null) throw new ArgumentNullException(nameof(allInputs));
			if (allLabels == null) throw new ArgumentNullException(nameof(allLabels));
			if (allInputs.Length != allLabels.Length) {
				throw new ArgumentException("Input count and label count must match.");
			}
			if (trainRatio <= 0.0 || trainRatio >= 1.0) {
				throw new ArgumentOutOfRangeException(nameof(trainRatio), "Train ratio must be between 0 and 1.");
			}

			int totalCount = allInputs.Length;
			int[] indices = new int[totalCount];
			for (int i = 0; i < totalCount; i++) {
				indices[i] = i;
			}

			// Fisher-Yates shuffle
			Random random = new Random(seed);
			for (int i = totalCount - 1; i > 0; i--) {
				int j = random.Next(i + 1);
				(indices[i], indices[j]) = (indices[j], indices[i]);
			}

			int trainCount = (int)(totalCount * trainRatio);
			int testCount = totalCount - trainCount;

			double[][] trainInputs = new double[trainCount][];
			int[] trainLabels = new int[trainCount];
			for (int i = 0; i < trainCount; i++) {
				int sourceIndex = indices[i];
				trainInputs[i] = (double[])allInputs[sourceIndex].Clone();
				trainLabels[i] = allLabels[sourceIndex];
			}

			double[][] testInputs = new double[testCount][];
			int[] testLabels = new int[testCount];
			for (int i = 0; i < testCount; i++) {
				int sourceIndex = indices[trainCount + i];
				testInputs[i] = (double[])allInputs[sourceIndex].Clone();
				testLabels[i] = allLabels[sourceIndex];
			}

			return new DatasetSplit(trainInputs, trainLabels, testInputs, testLabels);
		}

		private static string[] SplitCsvLine(string line) {
			string[] rawTokens = line.Split(';');
			for (int i = 0; i < rawTokens.Length; i++) {
				rawTokens[i] = Unquote(rawTokens[i].Trim());
			}
			return rawTokens;
		}

		private static string Unquote(string token) {
			if (token.Length >= 2 && token[0] == '"' && token[token.Length - 1] == '"') {
				return token.Substring(1, token.Length - 2);
			}
			return token;
		}

		private static double ParseDouble(string token, string columnName, int lineNumber) {
			if (!double.TryParse(token, NumberStyles.Float, CultureInfo.InvariantCulture, out double value)) {
				throw new InvalidDataException(
					$"Line {lineNumber}: invalid numeric value '{token}' in column '{columnName}'.");
			}
			return value;
		}

		private static int ParseBinary(string token, string columnName, int lineNumber) {
			if (string.Equals(token, "yes", StringComparison.OrdinalIgnoreCase)) {
				return 1;
			}
			if (string.Equals(token, "no", StringComparison.OrdinalIgnoreCase)) {
				return 0;
			}
			throw new InvalidDataException(
				$"Line {lineNumber}: invalid binary value '{token}' in column '{columnName}' (expected 'yes' or 'no').");
		}

		private static void WriteOneHot(
			double[] features,
			ref int offset,
			string token,
			string[] categories,
			string columnName,
			int lineNumber) {
			int matchedIndex = -1;
			for (int i = 0; i < categories.Length; i++) {
				if (string.Equals(token, categories[i], StringComparison.OrdinalIgnoreCase)) {
					matchedIndex = i;
					break;
				}
			}

			if (matchedIndex < 0) {
				throw new InvalidDataException(
					$"Line {lineNumber}: unknown category '{token}' in column '{columnName}'.");
			}

			features[offset + matchedIndex] = 1.0;
			offset += categories.Length;
		}
	}
}
