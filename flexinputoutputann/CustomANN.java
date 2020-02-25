/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package flexinputoutputann;

import java.util.ArrayList;

/**
 *
 * @author user
 */
public class CustomANN {

    InputLayer inputLayer;
    HiddenLayer hiddenLayer;
    OutputLayer outputLayer;

    double maxWeight = 5;
    double roundOffset = 0.5;
    double delta = 0.05;
    double changePercentage = 1;

    Data historicalData = new Data();

//    ArrayList<Link> allWeights;
    CustomANN() {

    }

    public double[] predict(Record input) {
        return outputNetwork(input);
    }

    public void addInputNode(String newNodeName) {
        Node node = new Node(newNodeName);
        Link link = new Link(null, node);
        link.weight = 1;
        node.inputLinks.add(link);
        for (int i = 0; i < hiddenLayer.nodes.size(); i++) {
            Link newLink = new Link(node, hiddenLayer.nodes.get(i));
            node.outputLinks.add(newLink);
            hiddenLayer.nodes.get(i).inputLinks.add(newLink);

//            Link newBiasLink = new Link(inputLayer.biasNode, hiddenLayer.nodes.get(i));
//            newBiasLink.data=1;
//            inputLayer.biasNode.outputLinks.add(newBiasLink);
        }
        inputLayer.nodes.add(node);

    }

    public void removeInputNode(String removingNodeName) {
        for (int i = 0; i < inputLayer.nodes.size(); i++) {
            if (inputLayer.nodes.get(i).name.equals(removingNodeName)) {
                for (int j = 0; j < hiddenLayer.nodes.size(); j++) {
                    hiddenLayer.nodes.get(j).inputLinks.remove(i);
                }
                for(int k=0;k<historicalData.inputRecords.size();k++)
                {
                    for(int m=0;m<historicalData.inputRecords.get(k).myData.size();m++)
                    {
                        if(historicalData.inputRecords.get(k).myData.get(m).name.equals(removingNodeName))
                        {
                            historicalData.inputRecords.get(k).myData.remove(m);
                        }
                    }
                }
                inputLayer.nodes.remove(i);
            }
        }
    }

    public void addOutputNode(String newNodeName) {
        Node node = new Node(newNodeName);
        for (int i = 0; i < hiddenLayer.nodes.size(); i++) {
            Link newLink = new Link(hiddenLayer.nodes.get(i), node);
            node.inputLinks.add(newLink);
            hiddenLayer.nodes.get(i).outputLinks.add(newLink);
        }
        Link newLink = new Link(hiddenLayer.biasNode, node);
        hiddenLayer.biasNode.outputLinks.add(newLink);
        node.inputLinks.add(newLink);
        Link link = new Link(node, null);
        link.weight = 1;
        node.outputLinks.add(link);
        outputLayer.nodes.add(node);
    }

    public void initTopology(int numInput, int numOutput, int hiddenLayerSize) {
        inputLayer = new InputLayer(hiddenLayer);
        hiddenLayer = new HiddenLayer(hiddenLayerSize, inputLayer, outputLayer);
        outputLayer = new OutputLayer(hiddenLayer);

    }

    private double[] outputNetwork(Record input) {
        for (int i = 0; i < input.myData.size(); i++) {
            for (int j = 0; j < inputLayer.nodes.size(); j++) {
                if (input.myData.get(i).name.equals(inputLayer.nodes.get(j).name)) {
                    inputLayer.nodes.get(j).inputLinks.get(0).data = input.myData.get(i).value;
                    break;
                }
            }
        }
        inputLayer.activateLayer();
        hiddenLayer.activateLayer();
        outputLayer.activateLayer();

        double output[] = new double[outputLayer.nodes.size()];
        for (int i = 0; i < outputLayer.nodes.size(); i++) {
            output[i] = outputLayer.nodes.get(i).outputLinks.get(0).data;
        }
        return output;
    }

    private double evalNetwork() {
        double error = 0;
        for (int r = 0; r < historicalData.inputRecords.size(); r++) {
            for (int i = 0; i < historicalData.inputRecords.get(r).myData.size(); i++) {
                for (int j = 0; j < inputLayer.nodes.size(); j++) {
                    if (historicalData.inputRecords.get(r).myData.get(i).name.equals(inputLayer.nodes.get(j).name)) {
                        inputLayer.nodes.get(j).inputLinks.get(0).data = historicalData.inputRecords.get(r).myData.get(i).value;
                        break;
                    }
                }
            }
            inputLayer.activateLayer();
            hiddenLayer.activateLayer();
            outputLayer.activateLayer();

            for (int k = 0; k < outputLayer.nodes.size(); k++) {
                if (outputLayer.nodes.get(k).name.equals(historicalData.targetRecords.get(r).myData.get(0).name)) {
                    double T = 1;
                    double O = partialRoundToZeroOne(outputLayer.nodes.get(k).outputLinks.get(0).data, roundOffset);
                    error += Math.abs(T - O);
                } else {
                    double T = 0;
                    double O = partialRoundToZeroOne(outputLayer.nodes.get(k).outputLinks.get(0).data, roundOffset);
                    error += Math.abs(T - O);
                }
            }
        }

        return error;
    }

    private void moveForward() {
        inputLayer.activateLayer();
        hiddenLayer.activateLayer();
        outputLayer.activateLayer();
//        System.out.println("Output: ");
//        for (int i = 0; i < outputLayer.nodes.size(); i++) {
//            System.out.println("O" + i + ": " + outputLayer.nodes.get(i).outputLinks.get(0).data);
//        }
    }

    private void updateNumericDerivative() {
        double initError = evalNetwork();
//        System.out.println("initError: " + initError);
        ArrayList allWeights = new ArrayList();
        for (int i = 0; i < hiddenLayer.nodes.size(); i++) {
            for (int in = 0; in < hiddenLayer.nodes.get(i).inputLinks.size(); in++) {
                allWeights.add(hiddenLayer.nodes.get(i).inputLinks.get(in));
            }
            for (int ou = 0; ou < hiddenLayer.nodes.get(i).outputLinks.size(); ou++) {
                allWeights.add(hiddenLayer.nodes.get(i).outputLinks.get(ou));
            }
            for (int bl = 0; bl < inputLayer.biasNode.outputLinks.size(); bl++) {
                allWeights.add(inputLayer.biasNode.outputLinks.get(bl));
            }
            for (int bl = 0; bl < hiddenLayer.biasNode.outputLinks.size(); bl++) {
                allWeights.add(hiddenLayer.biasNode.outputLinks.get(bl));
            }
        }
        changeWeights(allWeights, initError);
    }

    private void changeWeights(ArrayList<Link> input, double initError) {
        double currentError = initError;
        for (int i = 0; i < input.size(); i++) {
            double newError = changeWeight(input.get(i), currentError);
            currentError = newError;
        }
    }

    private double changeWeight(Link input, double initError) {
        double originalWeight = input.weight;
        double change = Math.max(input.weight * changePercentage, delta);
        double newError = initError;

//        if (originalWeight < maxWeight) {
//            input.weight = originalWeight + change;
//            double changeError = evalNetwork();
//            if (changeError >= initError) {
//                input.weight = originalWeight;
//            } else {
//                newError = changeError;
//                return newError;
//            }
//        }
//        if (originalWeight > -maxWeight) {
//            input.weight = originalWeight - change;
//            double changeError = evalNetwork();
//            if (changeError >= initError) {
//                input.weight = originalWeight;
//            } else {
//                newError = changeError;
//                return newError;
//            }
//        }
        int numRandomIterations = 15;
        for (int i = 0; i < numRandomIterations; i++) {
            input.weight = -maxWeight + Math.random() * 2 * maxWeight;
            double changeError = evalNetwork();
            if (changeError >= initError) {
                input.weight = originalWeight;
            } else {
                newError = changeError;
//                System.out.println("SUCCESS RANDOM WEIGHT");
                return newError;
            }
        }
        return newError;
    }

    private double partialRoundToZeroOne(double input, double offset) {
        return input;
//        if(input<0.5-offset)
//        {
//            return 0;
//        }else if(input>0.5+offset)
//        {
//            return 1;
//        }else{
//            return input;
//        }
    }

    private void updateBackPropagation(Record target) {
        backwardDelta(target);
        double nu = 100;
        double deltaW = 0;
        for (int hn = 0; hn < hiddenLayer.nodes.size(); hn++) {
            for (int hil = 0; hil < hiddenLayer.nodes.get(hn).inputLinks.size(); hil++) {
                deltaW = -nu * hiddenLayer.nodes.get(hn).delta * hiddenLayer.nodes.get(hn).inputLinks.get(hil).data;
//                System.out.println("deltaW: "+deltaW);
                hiddenLayer.nodes.get(hn).inputLinks.get(hil).weight = hiddenLayer.nodes.get(hn).inputLinks.get(hil).weight + deltaW;
            }
        }
    }

    private void backwardDelta(Record target) {
        for (int k = 0; k < outputLayer.nodes.size(); k++) {
            if (outputLayer.nodes.get(k).name.equals(target.myData.get(0).name)) {
                double T = 1;
                double O = outputLayer.nodes.get(k).outputLinks.get(0).data;
                outputLayer.nodes.get(k).delta = O * (1 - O) * (O - T);
            } else {
                double T = 0;
                double O = outputLayer.nodes.get(k).outputLinks.get(0).data;
                outputLayer.nodes.get(k).delta = O * (1 - O) * (O - T);
            }
        }
        for (int j = 0; j < hiddenLayer.nodes.size(); j++) {
            double O = hiddenLayer.nodes.get(j).outputLinks.get(0).data;
            double sum = 0;
            for (int k = 0; k < outputLayer.nodes.size(); k++) {
                sum = sum + outputLayer.nodes.get(k).inputLinks.get(j).weight * outputLayer.nodes.get(k).delta;
            }
            hiddenLayer.nodes.get(j).delta = O * (1 - O) * sum;
        }
    }

    public void inputData(Record input, Record target) {
        historicalData.inputRecords.add(input);
        historicalData.targetRecords.add(target);
        for (int i = 0; i < input.myData.size(); i++) {
            boolean isFeatureFound = false;
            for (int j = 0; j < inputLayer.nodes.size(); j++) {
                if (input.myData.get(i).name.equals(inputLayer.nodes.get(j).name)) {
//                    inputLayer.nodes.get(j).inputLinks.get(0).data = input.myData.get(i).value;
                    isFeatureFound = true;
                    break;
                }
            }
            if (isFeatureFound == false) {
                addInputNode(input.myData.get(i).name);
                inputLayer.nodes.get(inputLayer.nodes.size() - 1).inputLinks.get(0).data = input.myData.get(i).value;
            }
        }
        for (int i = 0; i < target.myData.size(); i++) {
            boolean isTargetFound = false;
            for (int j = 0; j < outputLayer.nodes.size(); j++) {
                if (target.myData.get(i).name.equals(outputLayer.nodes.get(j).name)) {
                    isTargetFound = true;
                    break;
                }
            }
            if (isTargetFound == false) {
                addOutputNode(target.myData.get(i).name);
            }
        }
//        System.out.println(outputLayer.nodes.get(0).outputLinks.get(0).data);
        double netError = evalNetwork();
        System.out.println("***");
        System.out.println("NetInitialError: " + netError);
        for (int i = 0; i < 20; i++) {
            updateNumericDerivative();
//            updateBackPropagation(target);
        }

//        System.out.println("Output: ");
//        for (int i = 0; i < outputLayer.nodes.size(); i++) {
//            System.out.println("O" + i + ": " + Math.round(outputLayer.nodes.get(i).outputLinks.get(0).data));
//        }
        netError = evalNetwork();
        System.out.println("NetFinalError: " + netError);
    }

}
