package classification;

import java.util.Map;
import java.util.Map.Entry;
import java.util.NavigableMap;
import java.util.Random;
import java.util.TreeMap;

public class WeightedRandom <T> {
  private final NavigableMap<Double, T> weights = new TreeMap<Double, T>();
  private final Random random = new Random();
  private double total = 0;
  
  public WeightedRandom(Map<T, Double> valuesAndWeights) {
    for (Entry<T, Double> entry: valuesAndWeights.entrySet()) {
      T val = entry.getKey();
      Double weight = entry.getValue();
      total += weight;
      weights.put(total, val);
    }
  }
  
  public T next() {
    double nextIndex = random.nextDouble() * total;
    return weights.ceilingEntry(nextIndex).getValue();
  }

}
