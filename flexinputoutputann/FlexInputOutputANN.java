/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package flexinputoutputann;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import weka.clusterers.SelfOrganizingMap;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;
import weka.core.converters.ConverterUtils.DataSource;

/**
 *
 * @author user
 */
public class FlexInputOutputANN {

    public static String dataLocalPath;

    FlexInputOutputANN() {
//        runManualANN();
        runWekaSOM();
    }

    public void runWekaSOM() {
        try {
            DataSource source = new DataSource("glass.arff");
            Instances data = source.getDataSet();
            SelfOrganizingMap SOM = new SelfOrganizingMap();
            SOM.setWidth(3);
            SOM.setHeight(3);
            SOM.buildClusterer(new Instances(data,0,1));
            for(int i=1;i<data.numInstances();i++)
            {
                SOM.updateClusterer(data.get(i));
            }
            System.out.println(SOM.m_clusterStats.length);
            System.out.println(SOM.m_clusterStats[0].length);
            System.out.println(SOM.m_clusterStats[0][0].length);
            System.out.println("***");
            for (int i = 0; i < SOM.m_clusterStats.length; i++) {
                System.out.println(SOM.header.get(i).name());
                for (int j = 0; j < SOM.m_clusterStats[i].length; j++) {
                    System.out.println();
                    for (int k = 0; k < SOM.m_clusterStats[i][j].length; k++) {
                        System.out.println(SOM.m_clusterStats[i][j][k]);
                    }
                }
            }
            System.out.println("Record in cluster");
            for (int i = 0; i < SOM.recordInClusters.length; i++) {
                System.out.println(SOM.recordInClusters[i]);
            }

        } catch (Exception ex) {
            Logger.getLogger(FlexInputOutputANN.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        try {
            DataSource source = new DataSource("glass.arff");
            Instances data = source.getDataSet();
            SelfOrganizingMap SOM = new SelfOrganizingMap();
            SOM.setWidth(3);
            SOM.setHeight(3);
            SOM.buildClusterer(data);
            System.out.println(SOM.m_clusterStats.length);
            System.out.println(SOM.m_clusterStats[0].length);
            System.out.println(SOM.m_clusterStats[0][0].length);
            System.out.println("***");
            for (int i = 0; i < SOM.m_clusterStats.length; i++) {
                System.out.println(SOM.header.get(i).name());
                for (int j = 0; j < SOM.m_clusterStats[i].length; j++) {
                    System.out.println();
                    for (int k = 0; k < SOM.m_clusterStats[i][j].length; k++) {
                        System.out.println(SOM.m_clusterStats[i][j][k]);
                    }
                }
            }
            System.out.println("Record in cluster");
            for (int i = 0; i < SOM.recordInClusters.length; i++) {
                System.out.println(SOM.recordInClusters[i]);
            }

        } catch (Exception ex) {
            Logger.getLogger(FlexInputOutputANN.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public void runManualANN() {
        int numRecordRepeat=6;
        CustomANN customANN = new CustomANN();
        customANN.initTopology(0, 0, 10);
        Record input = new Record();
        input.myData.add(new Datum("a", Datum.real, 1 + Math.random() / 100));
        Record target = new Record();
        target.myData.add(new Datum("t1", Datum.integer, 1));

        for (int i = 0; i < 4; i++) {
            input = new Record();
            input.myData.add(new Datum("a", Datum.real, 1 + Math.random() / 100));
            target = new Record();
            target.myData.add(new Datum("t1", Datum.integer, 1));
//            customANN.historicalData.inputRecords.add(input);
//            customANN.historicalData.targetRecords.add(target);
            customANN.inputData(input, target);
        }

        for (int i = 0; i < numRecordRepeat; i++) {
            input = new Record();
            input.myData.add(new Datum("b", Datum.real, 2 + Math.random() / 100));
            target = new Record();
            target.myData.add(new Datum("t1", Datum.integer, 1));
//            customANN.historicalData.inputRecords.add(input);
//            customANN.historicalData.targetRecords.add(target);
            customANN.inputData(input, target);
        }

        for (int i = 0; i < numRecordRepeat; i++) {
            input = new Record();
            input.myData.add(new Datum("c", Datum.real, 2 + Math.random() / 100));
            target = new Record();
            target.myData.add(new Datum("t1", Datum.integer, 1));
//            customANN.historicalData.inputRecords.add(input);
//            customANN.historicalData.targetRecords.add(target);
            customANN.inputData(input, target);
        }
        
        //****For t2
        for (int i = 0; i < numRecordRepeat; i++) {
            input = new Record();
            input.myData.add(new Datum("a", Datum.real, 3 + Math.random() / 100));
            target = new Record();
            target.myData.add(new Datum("t2", Datum.integer, 1));
//            customANN.historicalData.inputRecords.add(input);
//            customANN.historicalData.targetRecords.add(target);
            customANN.inputData(input, target);
        }

        for (int i = 0; i < numRecordRepeat; i++) {
            input = new Record();
            input.myData.add(new Datum("b", Datum.real, 4 + Math.random() / 100));
            target = new Record();
            target.myData.add(new Datum("t2", Datum.integer, 1));
//            customANN.historicalData.inputRecords.add(input);
//            customANN.historicalData.targetRecords.add(target);
            customANN.inputData(input, target);
        }

        for (int i = 0; i < numRecordRepeat; i++) {
            input = new Record();
            input.myData.add(new Datum("c", Datum.real, 5 + Math.random() / 100));
            target = new Record();
            target.myData.add(new Datum("t2", Datum.integer, 1));
//            customANN.historicalData.inputRecords.add(input);
//            customANN.historicalData.targetRecords.add(target);
            customANN.inputData(input, target);
        }

        for (int i = 0; i < numRecordRepeat; i++) {
//            input = new Record();
//            input.myData.add(new Datum("a", Datum.real, 1 + Math.random() / 100));
//            target = new Record();
//            target.myData.add(new Datum("t1", Datum.integer, 1));
////            customANN.historicalData.inputRecords.add(input);
////            customANN.historicalData.targetRecords.add(target);
//            customANN.inputData(input, target);
//
//            input = new Record();
//            input.myData.add(new Datum("a", Datum.real, 2 + Math.random() / 100));
//            target = new Record();
//            target.myData.add(new Datum("t2", Datum.integer, 1));
////            customANN.historicalData.inputRecords.add(input);
////            customANN.historicalData.targetRecords.add(target);
//            customANN.inputData(input, target);
//            
//            input = new Record();
//            input.myData.add(new Datum("a", Datum.real, 3 + Math.random() / 100));
//            target = new Record();
//            target.myData.add(new Datum("t3", Datum.integer, 1));
////            customANN.historicalData.inputRecords.add(input);
////            customANN.historicalData.targetRecords.add(target);
//            customANN.inputData(input, target);
//            
//            input = new Record();
//            input.myData.add(new Datum("a", Datum.real, 4 + Math.random() / 100));
//            target = new Record();
//            target.myData.add(new Datum("t4", Datum.integer, 1));
////            customANN.historicalData.inputRecords.add(input);
////            customANN.historicalData.targetRecords.add(target);
//            customANN.inputData(input, target);
        }
//        customANN.inputData(input, target);

//        for (int i = 0; i < 50; i++) {
//            input = new Record();
//            input.myData.add(new Datum("a", Datum.real, 3 + Math.random() / 100));
//            target = new Record();
//            target.myData.add(new Datum("t3", Datum.integer, 1));
//            customANN.inputData(input, target);
//        }
        input = new Record();
        input.myData.add(new Datum("a", Datum.real, 1 + Math.random() / 100));
        input.myData.add(new Datum("b", Datum.real, 2 + Math.random() / 100));
        input.myData.add(new Datum("c", Datum.real, 3 + Math.random() / 100));
        double[] prediction = customANN.predict(input);
        System.out.println("Prediction:");
        for (int i = 0; i < prediction.length; i++) {
            System.out.println("Prediction[" + i + "]: " + prediction[i]);
        }
        customANN.removeInputNode("b");
        
        input = new Record();
        input.myData.add(new Datum("a", Datum.real, 1 + Math.random() / 100));
        input.myData.add(new Datum("c", Datum.real, 3 + Math.random() / 100));
        prediction = customANN.predict(input);
        System.out.println("Prediction:");
        for (int i = 0; i < prediction.length; i++) {
            System.out.println("Prediction[" + i + "]: " + prediction[i]);
        }
        
        customANN.removeInputNode("c");
        
        input = new Record();
        input.myData.add(new Datum("a", Datum.real, 1 + Math.random() / 100));
        input.myData.add(new Datum("c", Datum.real, 3 + Math.random() / 100));
        prediction = customANN.predict(input);
        System.out.println("Prediction:");
        for (int i = 0; i < prediction.length; i++) {
            System.out.println("Prediction[" + i + "]: " + prediction[i]);
        }
//        customANN.inputData(input, target);
    }

    public void runDeepLearning4j() {
//        dataLocalPath = DownloaderUtility.CLASSIFICATIONDATA.Download();
//
//        //Load the training data:
//        RecordReader rr = new CSVRecordReader();
//        rr.initialize(new FileSplit(new File(dataLocalPath,"moon_data_train.csv")));
//        DataSetIterator trainIter = new RecordReaderDataSetIterator(rr,batchSize,0,2);
//
//        //Load the test/evaluation data:
//        RecordReader rrTest = new CSVRecordReader();
//        rrTest.initialize(new FileSplit(new File(dataLocalPath,"moon_data_eval.csv")));
//        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest,batchSize,0,2);
//        
        int seed = 123;
        double learningRate = 0.005;
        int numInputs = 2;
        int numOutputs = 2;
        int numHiddenNodes = 50;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(learningRate, 0.9))
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.SOFTMAX)
                        .nIn(numHiddenNodes).nOut(numOutputs).build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(100));    //Print score every 100 parameter updates

//        model.fit( trainIter, nEpochs );
//
//        System.out.println("Evaluate model....");
//        Evaluation eval = model.evaluate(testIter);
//
//        //Print the evaluation statistics
//        System.out.println(eval.stats());
    }

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // TODO code application logic here
        FlexInputOutputANN flexInputOutputANN = new FlexInputOutputANN();
    }

}
