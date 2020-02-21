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
public class Node {
    
    public ArrayList<Link> inputLinks=new ArrayList();
    public ArrayList<Link> outputLinks=new ArrayList();
    public String name;
    public double sumValue;
    
    public double delta;
    
    Node(String passed_name)
    {
        name=passed_name;
    }
    
    public void activateNode()
    {
        for(int i=0;i<outputLinks.size();i++)
        {
            outputLinks.get(i).data=checkActivationFcn();
        }
    }
    
    public double sumTheNode()
    {
        double summation=0;
        for(int i=0;i<inputLinks.size();i++)
        {
            summation=summation+inputLinks.get(i).data*inputLinks.get(i).weight;
        }
        sumValue=summation;
        return summation;
    }
    
    public double sigmoidValue(double input)
    {
        return -1+(1.0/(1.0+Math.exp(-input)))*2;
    }
    
    public double checkActivationFcn()
    {
        double sumValue=sumTheNode();
        return sigmoidValue(sumValue);
    }
    
}
