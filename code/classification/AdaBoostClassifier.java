package classification;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.logging.Logger;

public class AdaBoostClassifier implements Classifier {

  private static Logger logger = 
      Logger.getLogger(AdaBoostClassifier.class.getPackage().getName());
  
  private final Map<Classifier, Double> modelErrorRates = 
      new HashMap<Classifier, Double>();
  
  public AdaBoostClassifier(List<DataVector> trainingData, int numRounds) {     
    int numSamples = trainingData.size();
    
    //init tuple weights to 1 / numSamples
    Map<DataVector, Double> weights = new HashMap<DataVector, Double>();
    double initialWeight = ((double) 1) / numSamples;
    for (DataVector v: trainingData) {
      weights.put(v, initialWeight);
    }
    
    // build the composite model
    for (int k = 0; k < numRounds; k++) {
      // sample D with replacement based on weight
      List<DataVector> dataForRound = new ArrayList<DataVector>();
      WeightedRandom<DataVector> random = 
          new WeightedRandom<DataVector>(weights);
      for (int i = 0; i < numSamples; i++) {
        DataVector randomSelection = random.next();
        dataForRound.add(randomSelection);
      }
      
      double errorRate = 0;
      boolean[] isCorrect = new boolean[numSamples];
      do {
        // create model for this round
        NaiveBayesClassifier nb = new NaiveBayesClassifier(dataForRound);
        
        // calculate error rate 
        for (int n = 0; n < numSamples; n++) {
          DataVector v = dataForRound.get(n);
          String predicted = nb.classify(v);
          
          if (predicted.equals(v.getLabel())) {
            isCorrect[n] = true;
          }
          else {
            errorRate += weights.get(v);
          }
        }
        
        modelErrorRates.put(nb, errorRate);
        logger.fine(String.format("Round %d, error rate: %f", k, errorRate));
      } 
      while (errorRate > 0.5); // try again if the rate is too high
      
      double oldWeightsSum = sum(weights);
      
      //  multiply the weight of each correctly classified tuple 
      // by errR/(1-errR)
      double weightMultiplier = errorRate / (1 - errorRate);
      Set<DataVector> alreadyAdjusted = new HashSet<DataVector>();
      
      for (int n = 0; n < numSamples; n++) {
        DataVector vec = dataForRound.get(n);
        if (isCorrect[n] && !alreadyAdjusted.contains(vec)) {
          double oldWeight = weights.get(vec);
          double newWeight = oldWeight * weightMultiplier;
          weights.put(vec, newWeight);
          
          alreadyAdjusted.add(vec);
        }
      }
      
      double newWeightsSum = sum(weights);
      
      // normalize the weight of each tuple
      double multiplier = oldWeightsSum / newWeightsSum;
      for (Entry<DataVector, Double> entry: weights.entrySet()) {
        double weight = entry.getValue();
        weights.put(entry.getKey(), weight * multiplier);
      }
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
    
    return predictedLabel;
  }
}
