package test;

import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import org.junit.Test;

import classification.WeightedRandom;

/**
 * Not really a unit test. TODO come up with some way to test for correctness
 * of randomness.
 * 
 * @author pmarchwiak
 *
 */
public class WeightedRandomTest {

  @Test
  public void test1() {
    Map<String, Double> map = new HashMap<String, Double>();
    map.put("low", .3);
    map.put("medium", .7);
    map.put("medium2", .7);
    map.put("high", 1.3);
    
    WeightedRandom<String> random = new WeightedRandom<String>(map);
    
    Map<String, Integer> counts = new HashMap<String, Integer>();
    
    for (int i = 0; i < 100; i++) {
      String v = random.next();
      int count = 0;
      if (counts.containsKey(v)) {
        count = counts.get(v);
      }
      counts.put(v, count + 1);
    }
    
    for (Entry<String, Integer> entry: counts.entrySet()) {
      System.out.println(entry.getKey() + ": " + entry.getValue());
    }
  } 

}
