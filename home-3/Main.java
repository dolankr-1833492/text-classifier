import java.io.*;
import java.net.*;
import java.util.*;

import org.apache.commons.csv.*;

import smile.data.*;
import smile.io.*;

public class Main {
    public static void main(String[] args) throws IOException, URISyntaxException {
        DataFrame df = Read.csv("toxic.tsv", CSVFormat.TDF.withHeader());
        String[] corpus = df.stringVector("message").toStringArray();
        boolean[] labels = df.booleanVector("label").array();

        Random random = new Random(143);
        Vectorizer vectorizer = new BM25Vectorizer();
        Splitter splitter = new RandomSplitter(vectorizer.fitTransform(corpus), labels, random);
        TextClassifier clf = new TextClassifier(vectorizer, splitter);
        clf.print();

        int correct = 0;
        for (int i = 0; i < corpus.length; i += 1) {
            if (clf.classify(corpus[i]) == labels[i]) {
                correct += 1;
            }
        }
        System.out.println("Training accuracy: " + correct / (double) corpus.length);
    }
}
