package edu.kit.informatik.neuralnet;

import java.util.ArrayList;

public class NeuralNet {

  public ArrayList<Neuron[]> neuronLayers;

  public Neuron[]            inputLayer;
  public Neuron[]            outputLayer;
  
  private double learningRate = 0.2;

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

    // initalizing input layer
    inputLayer = new Neuron[layerSizes[0]];
    for (int i = 0; i < layerSizes[0]; i++) {
      inputLayer[i] = new Neuron();
    }
    neuronLayers.add(inputLayer);

    // initializing hidden/output layers
    for (int i = 1; i < layerSizes.length; i++) {
      Neuron[] neurons = new Neuron[layerSizes[i]];
      for (int j = 0; j < layerSizes[i]; j++)
        neurons[j] = new Neuron(neuronLayers.get(i - 1));
      neuronLayers.add(neurons);
    }
    
    outputLayer = this.neuronLayers.get(neuronLayers.size() - 1);
    
    // add upper layer for each neuron
    for (int i = 0; i < layerSizes.length - 1; i++) {
      for (Neuron n : neuronLayers.get(i))
        n.setUpperLayer(neuronLayers.get(i + 1));
    }
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

  public void process() {
    for (int i = 1; i < neuronLayers.size(); i++)
      processLayer(neuronLayers.get(i));
  }

  public void process(int steps) {
    for (int i = 0; i < steps; i++)
      this.process();
  }
  
  public void train(double[][] input, double[][] expected) {
    // train with dataset
  }
  
  public void updateInput(double[] input) {
    for (int i = 0; i < input.length; i++)
      this.inputLayer[i].setValue(input[i]);
  }
  
  //train with one data point
  public void train(double[] input, double[] expected) {
    // process the input
    if (this.inputLayer.length != input.length)
      return;
    updateInput(input);
    this.process();
    
    // error array
    if (expected.length != this.outputLayer.length)
      return;
    for (int i = 0; i < expected.length; i++)
      outputLayer[i].errorValue = this.outputLayer[i].getValue() - expected[i];
    
    // backpropagate node error
    for (int i = neuronLayers.size() - 2; i >= 0; i--) {
      for (int j = 0; j < neuronLayers.get(i).length; j++) {
        nodeError(i, j);
      }
    }
    
    // compute error on edges
    // over all layers
    for (int i = 1; i < neuronLayers.size(); i++) {
      // over all neurons
      for (int j = 0; j < neuronLayers.get(i).length; j++) {
        Neuron currentNeuron = neuronLayers.get(i)[j];
        double nodeSpecificError = currentNeuron.errorValue * Neuron.dSig(currentNeuron.getValue());
     
        // over all edges
        for (int k = 0; k < neuronLayers.get(i)[j].weights.length; k++) {
          currentNeuron.weights[k] -= learningRate * nodeSpecificError * neuronLayers.get(i - 1)[k].getValue();
        }
      }
    }
  }
  
  /**
   * Calculating the derivate of the error with respsect to this node.
   * Not usable with output nodes.
   * 
   * @return error value of the node
   */
  public double nodeError(int layer, int nodeNum) {
    double error = 0.0;
    
    Neuron node = neuronLayers.get(layer)[nodeNum];
    
    for (int i = 0; i < node.upperLayer.length; i++) {
      error += Neuron.dSig(node.upperLayer[i].getValue()) * node.upperLayer[i].weights[nodeNum] * node.upperLayer[i].errorValue;
    }
    
    // update error value for more efficient recursion
    node.errorValue = error;
    
    return error;
  }

  public static void main(String[] args) {
    // TODO Auto-generated method stub
    NeuralNet net = new NeuralNet(new int[] { 2, 3, 9 }, 0.5);
    
    for (int i = 0; i < 100000; i++) {
      net.train(new double[] { 0, 1 }, new double[] { 1, 0, 0, 1, 1, 1, 1, 1, 1 });

      System.out.println(net.outputLayer[0].getValue() + "..." + net.outputLayer[1].getValue());
    }
  }
}
