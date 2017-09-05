package edu.kit.informatik.neuralnet;

/**
 * A neuron in the neural net, knows about its lower layer.
 * 
 * @author 男子
 * @version 0.1
 */
public class Neuron {

  private int      position;
  private double   value;
  private double   errorValue; // this is the value for backpropagation
  public double[]  weights;

  private Neuron[] upperLayer;
  private Neuron[] lowerLayer;

  public Neuron(int position) {
    this.position = position;
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
   * Get the errValue.
   * 
   * @return the current error value of the neuron
   */
  public double getError() {
    return this.errorValue;
  }

  /**
   * Set the errValue.
   * 
   * @return the error value of the neuron
   */
  public void setError(double errorValue) {
    this.errorValue = errorValue;
  }

  /**
   * Set the value if it's in [0,1].
   * 
   * @param value
   *          the value to be set
   */
  public void setValue(double value) {
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
   * Set the lowerLayer and creating the correct weights array.
   * 
   * @param layer
   *          the value to be set
   */
  public void setLowerLayer(Neuron[] layer) {
    this.lowerLayer = layer;
    this.weights = new double[layer.length];
    for (int i = 0; i < lowerLayer.length; i++) {
      this.weights[i] = Math.random();
    }
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
   * Calculating and updating the derivate of the error with respsect to this
   * neuron.
   */
  public void updateError() {
    if (upperLayer == null)
      return;

    double error = 0.0;

    // calculate derivate of error based on this neuron
    for (int i = 0; i < this.upperLayer.length; i++) {
      error += Neuron.dSig(this.upperLayer[i].getValue()) * this.upperLayer[i].weights[position]
          * this.upperLayer[i].errorValue;
    }

    // update error value for more efficient recursion
    this.errorValue = error;
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

  /**
   * Actual derivate of sigmoid function.
   * 
   * @param value
   *          actual value
   * @return derivate of sigmoid of value
   */
  protected static double dSigmoid(double value) {
    return sigmoid(value) * (1 - sigmoid(value));
  }

  /**
   * Derivate of sigmoid given input is sigmoid(value)
   * 
   * @param value
   *          sigmoid(value) of value
   * @return derivative of sigmoid
   */
  protected static double dSig(double value) {
    return value * (1 - value);
  }
}
