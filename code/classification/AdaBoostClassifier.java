package classification;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.logging.Logger;

/**
 * AdaBoost using Naive Bayes.
 * @author pmarchwiak
 *
 */
public class AdaBoostClassifier implements Classifier {

  private static Logger logger = 
      Logger.getLogger(AdaBoostClassifier.class.getPackage().getName());
  
  private final Map<Classifier, Double> modelErrorRates = 
      new HashMap<Classifier, Double>();
  
  private static final int MAX_ATTEMPTS = -1;
  
  /**
   * 
   * @param trainingData
   * @param numRounds
   */
  public AdaBoostClassifier(List<DataVector> trainingData, int numRounds) {     
    int numSamples = trainingData.size();
    
    Map<DataVector, Double> weights = initializeWeights(trainingData);
    
    for (int k = 1; k <= numRounds; k++) {
      List<DataVector> dataForRound = randomSample(weights);
      
      double errorRate = 0;
      boolean[] isCorrect = new boolean[numSamples];
      NaiveBayesClassifier nb = null;
      int attempts = 0;
      do {
        nb = new NaiveBayesClassifier(dataForRound);
        
        errorRate = calcErrorRate(nb, trainingData, weights, isCorrect);
        
        logger.fine(String.format("Round %d, error rate: %f", k, errorRate));
        attempts++;
      } 
      while (errorRate > 0.5d && attempts != MAX_ATTEMPTS); // try again if the rate is too high
      
      if (attempts == MAX_ATTEMPTS) {
        System.out.println("Can't get error rate below 0.5 after " + 
            MAX_ATTEMPTS + " attempts.");
        System.exit(1);
      }
      
      double oldWeightsSum = sum(weights);
      
      adjustWeights(weights, dataForRound, isCorrect,
          errorRate);
      
      double newWeightsSum = sum(weights);
      
      normalizeWeights(weights, oldWeightsSum, newWeightsSum);
      
      modelErrorRates.put(nb, errorRate);
    }
  }

  /**
   * Init the weight of each tuple to 1/(numTuples)
   * @param trainingData
   * @return a map of tuples to their corresponding weight
   */
  private Map<DataVector, Double> initializeWeights(
      List<DataVector> trainingData) {
    Map<DataVector, Double> weights = new HashMap<DataVector, Double>();
    double initialWeight = ((double) 1) / trainingData.size();
    for (DataVector v: trainingData) {
      weights.put(v, initialWeight);
    }
    return weights;
  }
  
  /**
   * Sample according to tuple weights
   * @param weights map of tuples to weights
   * @return
   */
  private List<DataVector> randomSample(Map<DataVector, Double> weights) {
    List<DataVector> dataForRound = new ArrayList<DataVector>();
    WeightedRandom<DataVector> random = 
        new WeightedRandom<DataVector>(weights);
    for (int i = 0; i < weights.size(); i++) {
      DataVector randomSelection = random.next();
      dataForRound.add(randomSelection);
    }
    return dataForRound;
  }
  
  /**
   * Compute the error rate of the classifier:
   * error = sum(w_j * err(X_j)) where
   * w_j is the weight for tuple j,
   * err(X_j) is 1 when the tuple was misclassified, 0 otherwise
   * @param nb the classifier
   * @param data
   * @param weights
   * @param isCorrect array in which to store correctness for each tuple
   * 
   * @return the error rate of the classifier
   */
  private double calcErrorRate(NaiveBayesClassifier nb,
      List<DataVector> data, Map<DataVector, Double> weights,
      boolean[] isCorrect) {
    double errorRate = 0;
    for (int n = 0; n < data.size(); n++) {
      DataVector v = data.get(n);
      String predicted = nb.classify(v);
      
      if (predicted.equals(v.getLabel())) {
        isCorrect[n] = true;
      }
      else {
        errorRate += weights.get(v);
      }
    }
    return errorRate;
  }

  /**
   * For each tuple that was correctly classified, multiply the weight of the
   * tuple by (errorRate / (1 - errorRate))
   * @param weights
   * @param dataForRound
   * @param isCorrect
   * @param errorRate
   */
  private void adjustWeights(Map<DataVector, Double> weights,
      List<DataVector> dataForRound, boolean[] isCorrect,
      double errorRate) {
    double weightMultiplier = errorRate / (1 - errorRate);
    Set<DataVector> alreadyAdjusted = new HashSet<DataVector>();
    for (int n = 0; n < dataForRound.size(); n++) {
      DataVector vec = dataForRound.get(n);
      if (isCorrect[n] && !alreadyAdjusted.contains(vec)) {
        double oldWeight = weights.get(vec);
        double newWeight = oldWeight * weightMultiplier;
        weights.put(vec, newWeight);
        
        alreadyAdjusted.add(vec);
      }
    }
  }
  
  /**
   * Normalize weights by multiplying by the sum of the old weights and 
   * dividing by the sum of the new weights.
   * @param weights
   * @param oldWeightsSum
   * @param newWeightsSum
   */
  private void normalizeWeights(Map<DataVector, Double> weights,
      double oldWeightsSum, double newWeightsSum) {
    double multiplier = oldWeightsSum / newWeightsSum;
    for (Entry<DataVector, Double> entry: weights.entrySet()) {
      double weight = entry.getValue();
      weights.put(entry.getKey(), weight * multiplier);
    }
  }

  private double sum(Map<DataVector, Double> weights) {
    double sum = 0;
    for (Double weight: weights.values()) {
      sum += weight;
    }
    return sum;
  }

  @Override
  public String classify(DataVector vector) {
    Map<String, Double> labelVotes = new HashMap<String, Double>();
    
    for (Classifier model:  modelErrorRates.keySet()) {
      String label = model.classify(vector);
      
      Double votes = labelVotes.get(label);
      if (votes == null) {
        votes = Double.valueOf(0);
      }
      
      double errorRate =  modelErrorRates.get(model);
      double vote = Math.log((1 - errorRate) / errorRate);
      labelVotes.put(label, votes + vote);
    }
    
    double max = 0;
    String predictedLabel = "";
    for (String label: labelVotes.keySet()) {
      double votes = labelVotes.get(label);
      if (votes > max) {
        max = votes;
        predictedLabel = label;
      }
    }
    
    if (predictedLabel.length() == 0) {
      logger.warning("No label predicted for vector " + vector);
    }
    
    return predictedLabel;
  }
}
