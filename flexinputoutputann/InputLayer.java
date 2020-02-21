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
public class InputLayer {
    public ArrayList<Node> nodes;
    public HiddenLayer nextLayer;
    public Node biasNode=new Node("Bias");

    InputLayer(HiddenLayer passed_nextLayer) {
        nodes = new ArrayList();
        nextLayer = passed_nextLayer;
    }

    public void activateLayer() {
        for (int i = 0; i < nodes.size(); i++) {
            for(int j=0;j<nodes.get(i).outputLinks.size();j++)
            {
                nodes.get(i).outputLinks.get(j).data=nodes.get(i).inputLinks.get(0).data;
            }
        }
    }
}
