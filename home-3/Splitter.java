// Splitters provide a split method for dividing the given data points into left and right.
public interface Splitter {
    // Returns the best split and the left and right splitters, or null if no good split exists.
    public Result split();

    // Returns the majority label for this splitter.
    public boolean label();

    // Returns the number of data points in this splitter.
    public int size();

    // The left and right splitters that result from applying a split.
    public static class Result {
        public final Split split;
        public final Splitter left;
        public final Splitter right;

        protected Result(Split split, Splitter left, Splitter right) {
            this.split = split;
            this.left = left;
            this.right = right;
        }
    }
}
