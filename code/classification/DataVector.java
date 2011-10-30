package classification;

public class DataVector {
  private String label;
  private String[] data;
  
  public DataVector(String label, String[] data) {
    this.label = label;
    this.data = data;
  }

  public String getLabel() {
    return label;
  }

  public String[] getData() {
    return data;
  }
}
