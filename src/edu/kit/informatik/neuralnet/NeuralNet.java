package edu.kit.informatik.neuralnet;

import java.util.ArrayList;

/**
 * A Neural Net.
 * 
 * @author 男子, >Denis
 *
 */
public class NeuralNet {

  public ArrayList<Neuron[]> neuronLayers;

  public Neuron[]            inputLayer;
  public Neuron[]            outputLayer;

  private double             learningRate;

  /**
   * Create a neural net with learning rate and given layers.
   * 
   * @param layerSizes
   *          array countaining amount of neurons per layer (1 layer = 1 array
   *          entry)
   * @param learningRate
   *          learning rate
   */
  public NeuralNet(int[] layerSizes, double learningRate) {
    // Error handling illegal inputs
    if (learningRate < 0 || learningRate > 1)
      throw new IllegalArgumentException("learning rate must stay between 0 and 1");
    if (layerSizes.length < 3)
      throw new IllegalArgumentException("at least three layers needed in a neural net");
    for (int x : layerSizes) {
      if (x < 1)
        throw new IllegalArgumentException("layer sizes must be bigger then 0");
    }

    this.learningRate = learningRate;

    neuronLayers = new ArrayList<Neuron[]>();

    // initializing layers
    for (int i = 0; i < layerSizes.length; i++) {
      Neuron[] neurons = new Neuron[layerSizes[i]];
      for (int j = 0; j < layerSizes[i]; j++)
        neurons[j] = new Neuron(j);
      neuronLayers.add(neurons);
    }

    // adding connections to neurons
    for (int i = 0; i < layerSizes.length; i++) {
      for (Neuron n : this.neuronLayers.get(i)) {
        if (i != 0)
          n.setLowerLayer(this.neuronLayers.get(i - 1));
        if (i != layerSizes.length - 1)
          n.setUpperLayer(this.neuronLayers.get(i + 1));
      }
    }

    // for convenience ...
    inputLayer = this.neuronLayers.get(0);
    outputLayer = this.neuronLayers.get(neuronLayers.size() - 1);
  }

  /**
   * Processing a specific layer of the net.
   * 
   * @param layer
   *          the layer to be processed, except for the input layer!
   */
  private void processLayer(Neuron[] layer) {
    for (Neuron neuron : layer)
      neuron.processLowerLayer();
  }

  /**
   * Processing the input completely to the output once.
   */
  public void process() {
    for (int i = 1; i < neuronLayers.size(); i++)
      processLayer(neuronLayers.get(i));
  }

  /**
   * Updates the input layer with given values.
   * 
   * @param input
   *          the input, has to be the same then input layer size
   */
  public void updateInput(double[] input) {
    if (this.inputLayer.length != input.length)
      throw new IllegalArgumentException("input layer length not matched with training data");
    for (int i = 0; i < input.length; i++)
      this.inputLayer[i].setValue(input[i]);
  }

  /**
   * Train the net through backpropagation with one data set.
   * 
   * @param input
   *          the input, match with input layer
   * @param expected
   *          expected output, match with output layer
   */
  public void train(double[] input, double[] expected) {
    // process the input
    if (this.inputLayer.length != input.length)
      return;
    this.updateInput(input);
    this.process();

    // error array
    if (expected.length != this.outputLayer.length)
      return;
    for (int i = 0; i < expected.length; i++)
      outputLayer[i].setError(this.outputLayer[i].getValue() - expected[i]);

    // backpropagate neuron error
    for (int i = neuronLayers.size() - 2; i >= 0; i--) {
      for (int j = 0; j < neuronLayers.get(i).length; j++) {
        this.neuronLayers.get(i)[j].updateError();
      }
    }

    // compute error on edges
    // over all layers
    for (int i = 1; i < neuronLayers.size(); i++) {
      // over all neurons
      for (int j = 0; j < neuronLayers.get(i).length; j++) {
        Neuron currentNeuron = neuronLayers.get(i)[j];
        double nodeSpecificError = currentNeuron.getError() * Neuron.dSig(currentNeuron.getValue());

        // over all edges
        for (int k = 0; k < neuronLayers.get(i - 1).length; k++) {
          currentNeuron.weights[k] -= learningRate * nodeSpecificError
              * neuronLayers.get(i - 1)[k].getValue();
        }
      }
    }
  }

  /**
   * Train with dataset.
   * 
   * @param input
   *          dataset to be trained
   * @param expected
   *          expected values per dataset
   */
  public void train(double[][] input, double[][] expected) {
    for (int i = 0; i < input.length; i++) {
      this.train(input[i], expected[i]);
    }
  }

  /**
   * Train with dataset a specific amount of iterations.
   * 
   * @param input
   *          dataset to be trained
   * @param expected
   *          expected values per dataset
   * @param iterations
   *          how many iterations
   */
  public void train(double[][] input, double[][] expected, int iterations) {
    for (int i = 0; i < iterations; i++) {
      this.train(input, expected);
    }
  }

  /**
   * Testmethod...
   */
  public static void main(String[] args) {
    NeuralNet net = new NeuralNet(new int[] { 2, 3, 9 }, 0.5);

    for (int i = 0; i < 100000; i++) {
      net.train(new double[] { 0, 1 }, new double[] { 1, 0.5, 0, 1, 1, 1, 1, 1, 1 });

      System.out.println(net.outputLayer[0].getValue() + "..." + net.outputLayer[1].getValue());
    }
  }
}
