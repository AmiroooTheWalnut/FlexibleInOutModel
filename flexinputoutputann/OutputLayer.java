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
public class OutputLayer {
    public ArrayList<Node> nodes;
    public HiddenLayer previousLayer;
    public Node biasNode=new Node("Bias");

    OutputLayer(HiddenLayer passed_previousLayer) {
        nodes = new ArrayList();
        previousLayer = passed_previousLayer;
    }

    public void activateLayer() {
        for (int i = 0; i < nodes.size(); i++) {
            nodes.get(i).activateNode();
        }
//        for (int i = 0; i < nodes.size(); i++) {
//            for(int j=0;j<nodes.get(i).outputLinks.size();j++)
//            {
//                nodes.get(i).outputLinks.get(j).data=nodes.get(i).inputLinks.get(0).data;
//            }
//        }
    }
}
