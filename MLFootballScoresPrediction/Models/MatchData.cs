namespace MLFootballScoresPrediction.Models
{
    using Microsoft.ML.Data;
    public class MatchData
    {
        [LoadColumn(0)] public string HomeTeam { get; set; }
        [LoadColumn(1)] public string AwayTeam { get; set; }
        [LoadColumn(2)] public float Over2_5 { get; set; }
        [LoadColumn(3)] public float Under2_5 { get; set; }
        [LoadColumn(4)] public float HTGoals { get; set; } // Label source
    }
}
