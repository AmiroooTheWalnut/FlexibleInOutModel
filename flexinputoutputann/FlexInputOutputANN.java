/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package flexinputoutputann;

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

/**
 *
 * @author user
 */
public class FlexInputOutputANN {

    public static String dataLocalPath;

    FlexInputOutputANN() {
        runManualANN();
    }

    public void runManualANN() {
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

            input = new Record();
            input.myData.add(new Datum("a", Datum.real, 2 + Math.random() / 100));
            target = new Record();
            target.myData.add(new Datum("t2", Datum.integer, 1));
//            customANN.historicalData.inputRecords.add(input);
//            customANN.historicalData.targetRecords.add(target);
            customANN.inputData(input, target);
            
            input = new Record();
            input.myData.add(new Datum("a", Datum.real, 3 + Math.random() / 100));
            target = new Record();
            target.myData.add(new Datum("t3", Datum.integer, 1));
//            customANN.historicalData.inputRecords.add(input);
//            customANN.historicalData.targetRecords.add(target);
            customANN.inputData(input, target);
            
            input = new Record();
            input.myData.add(new Datum("a", Datum.real, 4 + Math.random() / 100));
            target = new Record();
            target.myData.add(new Datum("t4", Datum.integer, 1));
//            customANN.historicalData.inputRecords.add(input);
//            customANN.historicalData.targetRecords.add(target);
            customANN.inputData(input, target);
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
        input.myData.add(new Datum("a", Datum.real, 4 + Math.random() / 100));
        double[] prediction = customANN.predict(input);
        System.out.println("Prediction:");
        for(int i=0;i<prediction.length;i++)
        {
            System.out.println("Prediction["+i+"]: "+prediction[i]);
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
