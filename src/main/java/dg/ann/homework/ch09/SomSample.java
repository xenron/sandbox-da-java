package dg.ann.hw.ch09;

import geym.nn.som.ResultFrame;
import org.neuroph.core.Neuron;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.data.norm.Normalizer;
import org.neuroph.core.data.norm.RangeNormalizer;
import org.neuroph.nnet.Kohonen;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class SomSample {
    public static void main(String args[]) throws Exception {
        int INPUT_SIZE = 5;
        DataSet ds = new DataSet(INPUT_SIZE);
        double[][] data = new double[44][5];
        String[] dataKey = new String[44];
        List<String> records = readStringFromFile(new File("D:/tmp/city_gdp.txt"));
        for (int i = 1; i < records.size(); i++) {
            String record = records.get(i);
            String[] items = record.split(" ");
            double[] row = new double[INPUT_SIZE];
            for (int j = 0; j < INPUT_SIZE; j++) {
                row[j] = Double.parseDouble(items[j + 1].trim());
            }
            dataKey[i - 1] = items[0];
            data[i - 1] = row;
        }

        // 数据预处理
        // 大于等于平均值置为1，否则置为0
        for (int j=0;j<data[0].length;j++){
            double sum =0, avg=0;
            for (double[] row : data) sum += row[j];
            avg=sum/data.length;
            for (double[] row : data) {
                if (row[j]>=avg) row[j]=1; else row[j]=0;
            }
        }

//        Normalizer normalizer = new RangeNormalizer(0, 1);
//        normalizer.normalize(ds);
//
        ResultFrame frame = new ResultFrame();
        Kohonen som = new Kohonen(5, 100);
        for (double[] row : data) ds.addRow(new DataSetRow(row));

        som.learn(ds);

        for (int i=0;i<data.length;i++) {
            som.setInput(data[i]);
            som.calculate();
            int winnerIndex=getWinnerIndex(som);
            int x=getRowFromIndex(winnerIndex);
            int y=getColFromIndex(winnerIndex);
            System.out.println(dataKey[i]+" "+x+" "+y );
            frame.addElementString(new ResultFrame.ElementString(dataKey[i], x, y));
        }

        frame.showMe();

    }

    public static List<String> readStringFromFile(File file) throws FileNotFoundException, IOException {
        BufferedReader reader = null;
        List<String> list = new ArrayList<>();
        try {
            reader = new BufferedReader(new FileReader(file));
            String ex = "";

            while ((ex = reader.readLine()) != null) {
                list.add(ex);
            }
            return list;
        } catch (FileNotFoundException ex) {
            throw ex;
        } catch (IOException ex) {
            throw ex;
        } finally {
            if (reader != null) {
                reader.close();
            }
        }
    }

    // get unit with closetst weight vector
    private static int getWinnerIndex(Kohonen neuralNetwork) {
        Neuron winner = new Neuron();
        double minOutput = 100;
        int winnerIndex=-1;
        Neuron[] neurons=neuralNetwork.getLayerAt(1).getNeurons();
        for (int i=0;i<neurons.length;i++) {
            double out = neurons[i].getOutput();
            if (out < minOutput) {
                minOutput = out;
                winnerIndex = i;
            } // if
        } // while
        return winnerIndex;
    }

    /**
     * 10行10列中的位置
     * @param index
     * @return
     */
    private static int getRowFromIndex(int index){
        return index/10+1;
    }
    private static int getColFromIndex(int index){
        return index%10+1;
    }
}
