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
public class Datum {
    String name;
    String type;
    double value;
    public static String real="REAL";
    public static String integer="INTEGER";
    
    Datum(String passed_name, String passed_type, double passed_value)
    {
        name=passed_name;
        type=passed_type;
        value=passed_value;
    }
    
}
