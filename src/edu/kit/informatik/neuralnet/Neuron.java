package edu.kit.informatik.neuralnet;

/**
 * A neuron in the neural net, knows about its lower layer.
 * 
 * @author 男子
 * @version 0.1
 */
public class Neuron {

  private double   value;
  public double   errorValue; // this is the value for backpropagation
  public double[] weights;

  public Neuron[] upperLayer;
  private Neuron[] lowerLayer;

  /**
   * Lowest layer ( input layer ) constructor.
   */
  public Neuron() {
    this.value = 1.0;
    this.errorValue = 1.0;
  }

  /**
   * Constructor for Neurons in the hidden and output layer.
   * 
   * @param lowerLayer
   *          the layer beneath it
   */
  public Neuron(Neuron[] lowerLayer) {
    this.value = 0.0;
    this.weights = new double[lowerLayer.length];
    for (int i = 0; i < lowerLayer.length; i++) {
      this.weights[i] = Math.random();
    }
    this.lowerLayer = lowerLayer;
    this.errorValue = 1.0;
  }

  /**
   * Get the value.
   * 
   * @return the current value of the neuron
   */
  public double getValue() {
    return value;
  }

  /**
   * Set the value if it's in [0,1].
   * 
   * @param value
   *          the value to be set
   */
  public void setValue(double value) {
    if (value > 1 || value < 0)
      return;
    this.value = value;
  }

  /**
   * Set the upperLayer.
   * 
   * @param layer
   *          the value to be set
   */
  public void setUpperLayer(Neuron[] layer) {
    this.upperLayer = layer;
  }

  /**
   * Process the lower layer once, weighted sum addition and sigmoid.
   */
  public void processLowerLayer() {
    double weightedSum = 0.0;
    for (int i = 0; i < lowerLayer.length; i++) {
      weightedSum += lowerLayer[i].getValue() * weights[i];
    }

    this.value = sigmoid(weightedSum);
  }

  /**
   * Sigmoid function to normalize.
   * 
   * @param value
   *          number to be normalized
   * @return normalized number
   */
  protected static double sigmoid(double value) {
    return 1 / (1 + Math.pow(Math.E, -value));
  }

  protected static double dSigmoid(double value) {
    return sigmoid(value) * (1 - sigmoid(value));
  }
  
  protected static double dSig(double value) {
    return value * (1 - value);
  }
}
