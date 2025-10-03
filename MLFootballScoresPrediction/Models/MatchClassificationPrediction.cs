namespace MLFootballScoresPrediction.Models
{
    using Microsoft.ML.Data;

    /// <summary>
    /// Prediction output for classification with probability
    /// </summary>
    public class MatchClassificationPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool PredictedLabel { get; set; }

        [ColumnName("Probability")]
        public float Probability { get; set; } // Probability of the positive class (HTGoals > 0)

        [ColumnName("Score")]
        public float Score { get; set; } // Raw score before calibration
    }
}
