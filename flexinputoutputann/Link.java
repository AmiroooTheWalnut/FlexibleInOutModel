/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package flexinputoutputann;

/**
 *
 * @author user
 */
public class Link {
    public double weight=(Math.random()-0.5)/100.0;
    public double data;
    Node fromNode;
    Node toNode;
    
    public Link(Node passed_fromNode,Node passed_toNode)
    {
        fromNode=passed_fromNode;
        toNode=passed_toNode;
    }
}
