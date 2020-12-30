# Principal Component Analysis and Clustering
## 1. K-Means Clustering
### 1.1. Implementing K-Means

The K-means algorithm is a method to automatically cluster similar data examples together. Concretely, you are given a training set and want to group the data into a few cohesive "clusters". The intuition behind K-means is an iterative procedure that starts by guessing the initial centroids, and then refines this guess by repeatedly assigning examples to their closest centroids and then recomputing the centroids based on the assignments. The K-means algorithm is as follows:

    Initialize centroids

    centroids = kMeansInitCentroids(X, K);

    for iter = 1:iterations

      % Cluster assignment step: Assign each data point to the

      % closest centroid. idx(i) corresponds to cˆ(i), the index

      % of the centroid assigned to example i

      idx = findClosestCentroids(X, centroids);

      % Move centroid step: Compute means based on centroid

      % assignments

      centroids = computeMeans(X, idx, K);

    end

The inner-loop of the algorithm repeatedly carries out two steps: (i) Assigning each training example x(i) to its closest centroid, and (ii) Recomputing the mean of each centroid using the points assigned to it. The K-means algorithm will always converge to some final set of means for the centroids. Note that the converged solution may not always be ideal and depends on the initial setting of the centroids. Therefore, in practice the K-means algorithm is usually run a few times with different random initializations. One way to choose between these different solutions from different random initializations is to choose the one with the lowest cost function value (distortion).

### 1.2. Finding closest Centroids

In the \cluster assignment" phase of the K-means algorithm, the algorithm assigns every training example x(i) to its closest centroid, given the current positions of centroids. Specifically, for every example i we set 

                              c(i) := j that minimizes ||x(i) − µj||2;

where c(i) is the index of the centroid that is closest to x(i), and µj is the position (value) of the j’th centroid. Note that c(i) corresponds to idx(i) in the starter code. Our task is to complete the code in findClosestCentroids.m. This function takes the data matrix X and the locations of all centroids inside centroids and should output a one-dimensional array idx that holds the
index of the closest centroid to every training example. You can implement this using a loop over every training example and
every centroid. Once you have completed the code in findClosestCentroids.m, the script ex7.m will run oour code and we should see the output [1 3 2] corresponding to the centroid assignments for the first 3 examples.

### 1.3. Random Initialization

The initial assignments of centroids for the example dataset in ex7.m were designed so that we will see the same figure as in Figure 1. In practice, a good strategy for initializing the centroids is to select random examples from the training set.

    % Initialize the centroids to be random examples
    % Randomly reorder the indices of examples
    randidx = randperm(size(X, 1));
    % Take the first K examples as centroids
    centroids = X(randidx(1:K), :)
    
The code above first randomly permutes the indices of the examples (using randperm). Then, it selects the first K examples based on the random permutation of the indices. This allows the examples to be selected at random without the risk of selecting the same example twice.

## 2. Principal Component Analysis(PCA)
Principal component analysis (PCA) is a technique for reducing the dimensionality of such datasets, increasing interpretability but at the same time minimizing information loss. It does so by creating new uncorrelated variables that successively maximize variance. Finding such new variables, the principal components, reduces to solving an eigenvalue/eigenvector problem, and the new variables are defined by the dataset at hand, not a priori, hence making PCA an adaptive data analysis technique.

## 2.1. Implementing PCA
PCA consists of two computational steps: First, we compute the covariance matrix of the data. Then, we use Octave/MATLAB’s SVD function to compute the eigenvectors U1, U2, ....., Un. These will correspond to the principal components of variation in the data. Before using PCA, it is important to first normalize the data by subtracting the mean value of each feature from the dataset, and scaling each dimension so that they are in the same range. In the provided script ex7 pca.m,
this normalization has been performed using the featureNormalize function. After normalizing the data, you can run PCA to compute the principal components. Our task is to complete the code in pca.m to compute the principal components of the dataset. First, we should compute the covariance matrix of the data, which is given by:

                                       Σ = 1/m(X^T, X)
        
where X is the data matrix with examples in rows, and m is the number of examples. Note that Σ is a n × n matrix and not the summation operator. After computing the covariance matrix, we can run SVD on it to compute the principal components. In Octave/MATLAB, we can run SVD with the following command: [U, S, V] = svd(Sigma), where U will contain the principal components and S will contain a diagonal matrix.

## 2.2. Dimensionality Reduction with PCA
After computing the principal components, we can use them to reduce the feature dimension of our dataset by projecting each example onto a lower dimensional space, x(i) -> z(i) (e.g., projecting the data from 2D to 1D). We will use the eigenvectors returned by PCA and project the example dataset into a 1-dimensional space. In practice, if we are using a learning algorithm such as linear regression or perhaps neural networks, we could now use the projected data instead of the original data. By using the projected data, we can train your model faster as there are less dimensions in the input.

## 2.3. Projecting the data onto the principal components
We should now complete the code in projectData.m. Specifically, we are given a dataset X, the principal components U, and the desired number of dimensions to reduce to K. We should project each example in X onto the top K components in U. Note that the top K components in U are given by the first K columns of U, that is U reduce = U(:, 1:K). Once we have completed the code in projectData.m, ex7 pca.m will project the first example onto the first dimension and we should see a value of about 1.481 (or possibly -1.481, if you got −U1 instead of U1).
