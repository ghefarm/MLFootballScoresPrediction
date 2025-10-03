namespace MLFootballScoresPrediction.Models
{
    using Microsoft.ML.Data;

    public class MatchClassificationData
    {
        public string HomeTeam { get; set; }
        public string AwayTeam { get; set; }
        public float Over2_5 { get; set; }
        public float Under2_5 { get; set; }
        public float OddsDiff { get; set; } // Derived in later tries


        [ColumnName("Label")]
        public bool HTGoalsAbove0 { get; set; }
    }
}