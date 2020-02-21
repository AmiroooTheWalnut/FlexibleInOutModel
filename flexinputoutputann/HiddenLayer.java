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
public class HiddenLayer {

    public int size;
    public ArrayList<Node> nodes;
    public InputLayer previousLayer;
    public OutputLayer nextLayer;
    public Node biasNode=new Node("Bias");

    HiddenLayer(int passed_size, InputLayer passed_previousLayer, OutputLayer passed_nextLayer) {
        size = passed_size;
        nodes = new ArrayList();
        previousLayer = passed_previousLayer;
        nextLayer = passed_nextLayer;
        for (int i = 0; i < size; i++) {
            Node newNode = new Node("Hidden" + i);
            Link link=new Link(previousLayer.biasNode,newNode);
            link.data=1;
            previousLayer.biasNode.outputLinks.add(link);
            newNode.inputLinks.add(link);
            nodes.add(newNode);
        }
    }

    public void activateLayer() {
        for (int i = 0; i < nodes.size(); i++) {
            nodes.get(i).activateNode();
        }
    }
}
