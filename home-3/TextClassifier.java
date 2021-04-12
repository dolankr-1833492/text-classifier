//Katie Dolan
//Section AA
//This Text Classifier creates a classifier that will sort through text and decide whether it 
// should be flagged or not. What is flagged or not flagged is based on user input.
public class TextClassifier {
    //represents the top node of the tree.
    private Node overallRoot;
    //Vectorizor input from user.
    private Vectorizer vect;

    //pre: if splitter is empty the classifier wont be properly created, and of vectorizer is 
    //empty there won't be any way to classify texts.
    //post: constructs a TextClassifier instance from Vectorier vectorizer, and Splitter splitter.
    public TextClassifier(Vectorizer vectorizer, Splitter splitter) {
        vect = vectorizer;
        overallRoot = TextClassifierHelper(splitter);       
    }  

    //post: a helper that constructs the decision tree for classifying documents.
    private Node TextClassifierHelper(Splitter splitter) {
        Node root = new Node(splitter.label());
        Splitter.Result res = splitter.split();       
        if(res == null) {
            return root;
        } else {
            root = new Node(res.split, splitter.label(), TextClassifierHelper(res.left), 
            TextClassifierHelper(res.right));                
        }
        return root;
    }

    //post: returns the predicted label for the String using the TextClassifier.
    public boolean classify(String text) {
        double[] texts = vect.transform(text)[0];
        Node temp = overallRoot;
        return classify(texts, temp);       
    }  

    //helper that finds the predicted label for the text.
    private boolean classify(double[] texts, Node temp) {   

        if(temp.isLeaf()) {
            return temp.label;
        }
        else if(temp.split.goLeft(texts) == true) {
            return classify(texts, temp.left);
        }            
        else {
            return classify(texts, temp.right);
        }
    }

    //post: prints a decision tree is java code representation of the Text Classifier.
    public void print() {
        int spaces = 0;
        print(overallRoot, spaces);
    }

    //helper that runs through the tree and prints the appropriate java representations.
    private void print(Node root, int spaces) {
        if(root.isLeaf()) {           
                spacePrinter(spaces);
                System.out.println("return " + root.label + ";");                
        }
        else {   
            spacePrinter(spaces);    
            System.out.println("if (" + root.split.toString() + ")");
            print(root.left, spaces + 1);
            spacePrinter(spaces);
            System.out.println("else");
            print(root.right, spaces + 1);
        }
    }
    
    //helper that prints the amount of spaces neccessary based on how many indents there would be.
    private void spacePrinter(int spaces) {
        for(int i = spaces; i>0; i--) {
            System.out.print(" ");
        }
    }

    //post: prunes the decision tree to the given depth and reassigns the nodes at that depth to 
    //leaf nodes with the majority label.
    public void prune(int depth) {
        overallRoot = prune(overallRoot, depth);
    }

    //helper that finds the nodes at the given depth and reassigns them to leaf nodes.
    private Node prune(Node root, int depth) {
        if (root != null) {
            if(depth == 0) {
                root = new Node(root.label);      
            }
            else {
                root.left = prune(root.left, depth - 1);
                root.right = prune(root.right, depth - 1);
            }
        }
        return root;
    }

    // An internal node or a leaf node in the decision tree.
    private static class Node {
        public Split split;
        public boolean label;
        public Node left;
        public Node right;

        // Constructs a new leaf node with the given label.
        public Node(boolean label) {
            this(null, label, null, null);
        }

        // Constructs a new internal node with the given split, label, and left and right nodes.
        public Node(Split split, boolean label, Node left, Node right) {
            this.split = split;
            this.label = label;
            this.left = left;
            this.right = right;
        }

        // Returns true if and only if this node is a leaf node.
        public boolean isLeaf() {
            return left == null && right == null;
        }
    }
}
