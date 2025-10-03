namespace MLFootballScoresPrediction
{
    using Microsoft.ML;
    using Microsoft.ML.AutoML;
    using Microsoft.ML.Data;
    using Microsoft.ML.Trainers;
    using Microsoft.ML.Transforms;
    using MLFootballScoresPrediction.Models;

    public static class FootballTrainer
    {
        /// <summary>
        /// Trains three classification models for predicting whether any goals will be scored in the first half of Premier League football matches.
        /// </summary>
        /// <param name="mlContext"></param>
        public static void Train(MLContext mlContext)
        {
            string dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "football-data-2024.csv");
            IDataView data = mlContext.Data.LoadFromTextFile<MatchData>(path: dataPath, hasHeader: true, separatorChar: ',');

            var classificationData = mlContext.Data.CreateEnumerable<MatchData>(data, reuseRowObject: false)
                .Select(x => new MatchClassificationData
                {
                    HomeTeam = x.HomeTeam,
                    AwayTeam = x.AwayTeam,
                    Over2_5 = x.Over2_5,
                    Under2_5 = x.Under2_5,
                    OddsDiff = x.Over2_5 - x.Under2_5,
                    HTGoalsAbove0 = x.HTGoals > 0
                });

            IDataView classificationDataView = mlContext.Data.LoadFromEnumerable(classificationData);

            TrainCalssificationModelusingSGD(mlContext,classificationDataView);
            Console.WriteLine();
            TrainCalssificationModelUsingAP(mlContext,classificationDataView);
            Console.WriteLine();
            TrainCalssificationModelUsingAutoML(mlContext,classificationDataView);
            Console.WriteLine();
        }

        private static void TrainCalssificationModelusingSGD(MLContext mlContext, IDataView classificationDataView)
        {
            var pipeline = mlContext.Transforms.Categorical.OneHotEncoding(new[]
                {
                  new InputOutputColumnPair("HomeTeamEncoded", "HomeTeam"),
                  new InputOutputColumnPair("AwayTeamEncoded", "AwayTeam")
                })
                .Append(mlContext.Transforms.Concatenate("Features",
                  "HomeTeamEncoded", "AwayTeamEncoded", "Over2_5", "Under2_5", "OddsDiff"))
               .Append(mlContext.BinaryClassification.Trainers.SgdCalibrated());


            var trainTestSplit = mlContext.Data.TrainTestSplit(classificationDataView, testFraction: 0.2);
            var model = pipeline.Fit(trainTestSplit.TrainSet);
            var predictions = model.Transform(trainTestSplit.TestSet);
            
            Console.WriteLine("SGD: First-Try Metrics:");
            var metrics = mlContext.BinaryClassification.Evaluate(predictions);
            Console.WriteLine($"Accuracy: {(metrics.Accuracy * 100):F2} %");
            Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:F2}");
            Console.WriteLine($"F1 Score: {metrics.F1Score:F2}");

            Console.WriteLine($"Manual evaluation...");
            var predProbabilities = mlContext.Data.CreateEnumerable<MatchClassificationPrediction>(predictions, reuseRowObject: false);
            foreach (var predProbability in predProbabilities.Take(10))
            {
                Console.WriteLine($"Prediction: {predProbability.PredictedLabel}, Probability: {predProbability.Probability}");
            }

            mlContext.Model.Save(model, classificationDataView.Schema, "SGD_model.zip");

        }

        private static void TrainCalssificationModelUsingAP(MLContext mlContext, IDataView classificationDataView)
        {
            var pipeline = mlContext.Transforms.Categorical.OneHotEncoding(new[]
                {
                  new InputOutputColumnPair("HomeTeamEncoded", "HomeTeam"),
                  new InputOutputColumnPair("AwayTeamEncoded", "AwayTeam")
                })
                .Append(mlContext.Transforms.Concatenate("Features",
                  "HomeTeamEncoded", "AwayTeamEncoded", "Over2_5", "Under2_5", "OddsDiff"))
               .Append(mlContext.BinaryClassification.Trainers.AveragedPerceptron())
               .Append(mlContext.BinaryClassification.Calibrators.Platt());


            var trainTestSplit = mlContext.Data.TrainTestSplit(classificationDataView, testFraction: 0.2);
            var model = pipeline.Fit(trainTestSplit.TrainSet);
            var predictions = model.Transform(trainTestSplit.TestSet);

            Console.WriteLine("AveragedPerceptron: First-Try Metrics:");
            var metrics = mlContext.BinaryClassification.Evaluate(predictions);
            Console.WriteLine($"Accuracy: {(metrics.Accuracy * 100):F2} %");
            Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:F2}");
            Console.WriteLine($"F1 Score: {metrics.F1Score:F2}");

            Console.WriteLine($"Manual evaluation...");
            var predProbabilities = mlContext.Data.CreateEnumerable<MatchClassificationPrediction>(predictions, reuseRowObject: false);
            foreach (var predProbability in predProbabilities.Take(10))
            {
                Console.WriteLine($"Prediction: {predProbability.PredictedLabel}, Probability: {predProbability.Probability}");
            }

            mlContext.Model.Save(model, classificationDataView.Schema, "AP_model.zip");

        }

        private static void TrainCalssificationModelUsingAutoML(MLContext mlContext, IDataView classificationDataView)
        {
            var pipeline = mlContext.Transforms.Categorical.OneHotEncoding(new[]
                {
                  new InputOutputColumnPair("HomeTeamEncoded", "HomeTeam"),
                  new InputOutputColumnPair("AwayTeamEncoded", "AwayTeam")
                })
                .Append(mlContext.Transforms.Concatenate("Features",
                  "HomeTeamEncoded", "AwayTeamEncoded", "Over2_5", "Under2_5", "OddsDiff"))
               .Append(mlContext.BinaryClassification.Trainers.SgdCalibrated());

            // Create and run AutoML experiment
            var experimentSettings = new BinaryExperimentSettings
            {
                UseAutoZeroTuner = true,
                MaxExperimentTimeInSeconds = 150,// Try 600 (10 minutes) for more thorough search
                OptimizingMetric = BinaryClassificationMetric.AreaUnderRocCurve,
                MaxModels = 10 //Try 50 for more models, but increases time
            };

            var experiment = mlContext.Auto().CreateBinaryClassificationExperiment(experimentSettings);

            var trainTestSplit = mlContext.Data.TrainTestSplit(classificationDataView, testFraction: 0.2);
            var result = experiment.Execute(trainTestSplit.TrainSet, labelColumnName: "Label", preFeaturizer: pipeline);

            // Output best model results
            var bestRun = result.BestRun;
            Console.WriteLine($"Best Trainer: {bestRun.TrainerName}");
            Console.WriteLine($"Accuracy: {bestRun.ValidationMetrics.Accuracy}");
            Console.WriteLine($"AUC: {bestRun.ValidationMetrics.AreaUnderRocCurve}");
            Console.WriteLine($"AUPRC: {bestRun.ValidationMetrics.AreaUnderPrecisionRecallCurve}");
            Console.WriteLine($"F1: {bestRun.ValidationMetrics.F1Score}");

            Console.WriteLine("AutoML-Best-Run metrics on test data:");
            var predictions = bestRun.Model.Transform(trainTestSplit.TestSet);
            var metrics = mlContext.BinaryClassification.Evaluate(predictions);
            Console.WriteLine($"Accuracy: {metrics.Accuracy * 100:F2} %");
            Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:F2}");
            Console.WriteLine($"AUPRC: {metrics.AreaUnderPrecisionRecallCurve:F2}");
            Console.WriteLine($"F1 Score: {metrics.F1Score:F2}");

            mlContext.Model.Save(bestRun.Model, classificationDataView.Schema, "AutoML_model.zip");
        }

        /// <summary>
        /// Predicts whether any goals will be scored in the first half of Premier League football matches using three trained classification models.
        /// </summary>
        public static string Predict(MLContext mlContext, MatchData input)
        {
            var example = new MatchClassificationData
            {
                HomeTeam = input.HomeTeam,
                AwayTeam = input.AwayTeam,
                Over2_5 = input.Over2_5,
                Under2_5 = input.Under2_5,
                OddsDiff = input.Over2_5 - input.Under2_5
            };

            if (!File.Exists("AP_model.zip"))
            {
                return "Please train the models first!";
            }
            
            ITransformer model = mlContext.Model.Load("AP_model.zip", out _);
            var predEngine = mlContext.Model.CreatePredictionEngine<MatchClassificationData, MatchClassificationPrediction>(model);
            var prediction = predEngine.Predict(example);
            var resultAP = $"AveragedPerceptron: Goals in first half? {prediction.PredictedLabel} (Probability: {prediction.Probability:F2})";

            model = mlContext.Model.Load("SGD_model.zip", out _);
            predEngine = mlContext.Model.CreatePredictionEngine<MatchClassificationData, MatchClassificationPrediction>(model);
            prediction = predEngine.Predict(example);
            var resultSGD = $"SGD: Goals in first half? {prediction.PredictedLabel} (Probability: {prediction.Probability:F2})";

            model = mlContext.Model.Load("AutoML_model.zip", out _);
            predEngine = mlContext.Model.CreatePredictionEngine<MatchClassificationData, MatchClassificationPrediction>(model);
            prediction = predEngine.Predict(example);
            var resultAutoML = $"AutoML_model: Goals in first half? {prediction.PredictedLabel} (Probability: {prediction.Probability:F2})";
            
            return $"{resultAP}{Environment.NewLine}{resultSGD}{Environment.NewLine}{resultAutoML}";
        }
    }
}
