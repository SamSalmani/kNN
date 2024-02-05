
#k-Nearest Neighburs (kNN)
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.stats as ss
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

# calc distance between two points
def distance(p1,p2):
    """ Calc distance between two points"""
    return np.sqrt(np.sum(np.power((p1-p2),2)))



# 1-a func called majority vote. Select the value that is occured the most randomly-better function as it gives the mode randomly
def majority_vote(votes):
    """Return the most common element in votes (mode)"""
    vote_counts={}
    for vote in votes:
        if vote in vote_counts:
            vote_counts[vote] +=1
        else:
            vote_counts[vote] = 1
    
    winners = []
    max_count = max(vote_counts.values())
    for vote,count in vote_counts.items():
        if count == max_count:
            winners.append(vote)
    
    return random.choice(winners)



# 2-a func called majority vote. Select the value that is occured the most using 
#mode function in scipy but it just gives one of them not randomly changed number between modes
def majority_vote_short(votes):
    """Return the most common element in votes (mode)"""
    mode,count = ss.mstats.mode(votes)
    return mode

votes = [1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,3,3,2,1,1,1,1,1,1,3,3,3,3]
majority_vote_short(votes)



'''HOW TO IMPLEMENT kNN (including pseudo code)
- loop oveer all points
- compute the distance between p and every other points
- sort distance and those points the are nearest to point p'''

def find_nearest_neighbors(p, points, k=5):
    """ Find the k nearest neighbours of point p and return the indices"""
    distances=np.zeros(points.shape[0])
    for i in range(len(distances)):
        distances[i] = distance(p, points[i])
    ind = np.argsort(distances) # np.argsort : rerurns the indices that would sort the given array
    return ind[:k]
   
points = np.array([[1,1],[1,2],[1,3],[2,1],[2,2],[2,3],[3,1],[3,2],[3,3]]) 
p=np.array([2.5,2])
ind = find_nearest_neighbors(p, points, 2); print(points[ind])

def knn_predict(p, points, outcomes, k=5): #outcomes are the classes
    """ predict the class of a point using kNN"""
    #find k nearest neighbors
    ind = find_nearest_neighbors(p, points, k)
    # predict the class (category) of p based on majority vote
    return majority_vote(outcomes[ind])

outcomes = np.array([0,0,0,0,1,1,1,1,1]) # these are the classes assigned to each point in points. It means each of those 9 points are reated to one the classes 1 or 0
knn_predict(np.array([2.5,2.7]), points, outcomes, k=2)
   
 
## generate synthetics data
def generate_synth_data(n=50):
    """Create two sets of points from bivariate normal distributions"""
    points = np.concatenate((ss.norm(0,1).rvs((n,2)), ss.norm(0,1).rvs((n,2))), axis=0) #concat two 2,5 arrays along rows to get a 2,10
    outcomes = np.concatenate((np.repeat(0,n), np.repeat(1,n)), axis=0)
    return (points, outcomes)


n=20
(points, outcomes) = generate_synth_data(n)
plt.figure()
plt.plot(points[:n,0], points[:n,1], "ro", label = "class 0")
plt.plot(points[n:,0], points[n:,1], "bo", label = "class 1")
plt.savefig("bivardata.pdf")


## Plot a Prediction Grid
def make_prediction_grid(predictors, outcomes, limits, h, k):
    
    """Classify each point on the prediction grid"""
    
    (x_min, x_max, y_min, y_max) = limits
    xs = np.arange(x_min, x_max, h)
    ys = np.arange(y_min, y_max, h)
    xx, yy = np.meshgrid(xs, ys)
    
    prediction_grid = np.zeros(xx.shape, dtype=int) # as class number is either 0 or 1, the data type is int
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            p = np.array([x,y])
            prediction_grid[j,i] = knn_predict(p, predictors, outcomes, k) #we assigned [j,i] as we want to assign y-velues(j) to rows and x-values to columns
            
    return (xx,yy,prediction_grid)

def plot_prediction_grid (xx, yy, prediction_grid, filename):
    """ Plot KNN predictions for every point on the grid."""
    from matplotlib.colors import ListedColormap
    background_colormap = ListedColormap (["hotpink","lightskyblue", "yellowgreen"])
    observation_colormap = ListedColormap (["red","blue","green"])
    plt.figure(figsize =(10,10))
    plt.pcolormesh(xx, yy, prediction_grid, cmap = background_colormap, alpha = 0.5)
    plt.scatter(predictors[:,0], predictors [:,1], c = outcomes, cmap = observation_colormap, s = 50)
    plt.xlabel('Variable 1'); plt.ylabel('Variable 2')
    plt.xticks(()); plt.yticks(())
    plt.xlim (np.min(xx), np.max(xx))
    plt.ylim (np.min(yy), np.max(yy))
    plt.savefig(filename)



(predictors, outcomes) = generate_synth_data()
k=50; filename = "knn_synth_5.pdf"; limits = (-3,4,-3,4); h=0.1
(xx,yy,prediction_grid) = make_prediction_grid(predictors, outcomes, limits, h, k)
plot_prediction_grid(xx,yy, prediction_grid, filename)



""" choosing the best k value that is not too large or too small is important. We use bias-variance trade off to know what is the best value"""

# Apply kNN to a real case study and compare it to scikit-learn kNN

iris = datasets.load_iris()

predictors = iris.data[:, 0:2]
outcomes = iris.target

plt.plot(predictors[outcomes==0][:,0], predictors[outcomes==0][:,1], "ro")
plt.plot(predictors[outcomes==1][:,0], predictors[outcomes==1][:,1], "go")
plt.plot(predictors[outcomes==2][:,0], predictors[outcomes==2][:,1], "bo")
plt.savefig("iris.pdf")

k=50; filename = "iris_grid.pdf"; limits = (4,8,1.5,4.4); h=0.1
(xx,yy,prediction_grid) = make_prediction_grid(predictors, outcomes, limits, h, k)
plot_prediction_grid(xx,yy, prediction_grid, filename)

my_predictions =  np.array([knn_predict(p, predictors, outcomes, 5) for p in predictors])

# scikit learn module
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(predictors, outcomes)
sk_predictions = knn.predict(predictors)

# compare our function with sklearn
print(np.mean(my_predictions == sk_predictions)*100)
print(np.mean(my_predictions == outcomes)*100)
print(np.mean(sk_predictions == outcomes)*100)







