using Microsoft.ML;
using MLFootballScoresPrediction;
using MLFootballScoresPrediction.Models;


MLContext mlContext = new(seed: 0);
Console.WriteLine("Train/Retrain models (y,n)?");
var doTrain = Console.ReadLine();
if (doTrain?.ToString().Equals("y", StringComparison.CurrentCultureIgnoreCase) ?? false)
{
    FootballTrainer.Train(mlContext);
}

var example = new MatchData
{
    HomeTeam = "Man City",
    AwayTeam = "Burnley",
    Over2_5 = 4.20f,
    Under2_5 = 1.18f
};

Console.WriteLine(FootballTrainer.Predict(mlContext, example));
Console.ReadLine();
