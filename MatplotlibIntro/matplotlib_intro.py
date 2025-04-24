# matplotlib_intro.py
"""Python Essentials: Intro to Matplotlib.
Noah Pettinato
MTH 420
4/24/2025
"""

import numpy as np
from matplotlib import pyplot as plt

# Problem 1
def var_of_means(n):
    """ Create an (n x n) array of values randomly sampled from the standard
    normal distribution. Compute the mean of each row of the array. Return the
    variance of these means.

    Parameters:
        n (int): The number of rows and columns in the matrix.

    Returns:
        (float) The variance of the means of each row.
    """
    samples = np.random.normal(size=(n, n))
    row_means = np.mean(samples, axis=1)
    return np.var(row_means)

def prob1():
    """ Create an array of the results of var_of_means() with inputs
    n = 100, 200, ..., 1000. Plot and show the resulting array.
    """
    n_values = np.arange(100, 1001, 100)
    variances = np.array([var_of_means(n) for n in n_values])
    
    plt.plot(variances)
    plt.title("Variance of Row Means vs Matrix Size Index")
    plt.xlabel("Index (0 = n=100, 1 = n=200, ... , 9 = n=1000)")
    plt.ylabel("Variance of Row Means")
    
    #plt.savefig("prob1_plot.png")
    #print("Plot saved to 'prob1_plot.png'")

    plt.show()


# Problem 2
def prob2():
    """ Plot the functions sin(x), cos(x), and arctan(x) on the domain
    [-2pi, 2pi]. Make sure the domain is refined enough to produce a figure
    with good resolution.
    """
    plt.clf()
    x = np.linspace(-2 * np.pi, 2* np.pi, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.arctan(x)
    
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.plot(x, y3)
    
    plt.title("sin(x), cos(x), arctan(x) on [-2π, 2π]")
    plt.xlabel("x")
    plt.ylabel("Function Values")
    
    plt.legend(["sin(x)", "cos(x)", "arctan(x)"])
    
    #plt.savefig("prob2_plot.png")
    #print("Plot saved to 'prob2_plot.png'")

    plt.show()
    
# Problem 3
def prob3():
    """ Plot the curve f(x) = 1/(x-1) on the domain [-2,6].
        1. Split the domain so that the curve looks discontinuous.
        2. Plot both curves with a thick, dashed magenta line.
        3. Set the range of the x-axis to [-2,6] and the range of the
           y-axis to [-6,6].
    """
    plt.clf()
    x1 = np.linspace(-2, 0.99, 500)
    x2 = np.linspace(1.01, 6, 500)
    
    y1 = 1 / (x1 - 1)
    y2 = 1 / (x2 - 1)
    
    plt.plot(x1, y1, 'm--', linewidth=4)
    plt.plot(x2, y2, 'm--', linewidth=4)
    
    plt.xlim(-2, 6)
    plt.ylim(-6, 6)
    
    
    #plt.savefig("prob3_plot.png")
    #print("Plot saved to 'prob3_plot.png'")

    plt.show()
    

# Problem 4
def prob4():
    """ Plot the functions sin(x), sin(2x), 2sin(x), and 2sin(2x) on the
    domain [0, 2pi], each in a separate subplot of a single figure.
        1. Arrange the plots in a 2 x 2 grid of subplots.
        2. Set the limits of each subplot to [0, 2pi]x[-2, 2].
        3. Give each subplot an appropriate title.
        4. Give the overall figure a title.
        5. Use the following line colors and styles.
              sin(x): green solid line.
             sin(2x): red dashed line.
             2sin(x): blue dashed line.
            2sin(2x): magenta dotted line.
    """
    plt.clf()
    x = np.linspace(0, 2 * np.pi, 500)
    fig, axes = plt.subplots(2, 2)
    fig.suptitle("Variations of Sine Functions", fontsize=16)
    
    axes[0, 0].plot(x, np.sin(x), 'g-') 
    axes[0, 0].set_title("sin(x)")
    axes[0, 0].axis([0, 2 * np.pi, -2, 2])

    axes[0, 1].plot(x, np.sin(2 * x), 'r--') 
    axes[0, 1].set_title("sin(2x)")
    axes[0, 1].axis([0, 2 * np.pi, -2, 2])

    axes[1, 0].plot(x, 2 * np.sin(x), 'b--')  
    axes[1, 0].set_title("2sin(x)")
    axes[1, 0].axis([0, 2 * np.pi, -2, 2])

    axes[1, 1].plot(x, 2 * np.sin(2 * x), 'm:')  
    axes[1, 1].set_title("2sin(2x)")
    axes[1, 1].axis([0, 2 * np.pi, -2, 2])

    #plt.subplots_adjust(hspace=0.5)
    #plt.savefig("prob4_plot.png")
    #print("Plot saved to 'prob4_plot.png'")

    plt.show()



# Problem 5
def prob5():
    """ Visualize the data in FARS.npy. Use np.load() to load the data, then
    create a single figure with two subplots:
        1. A scatter plot of longitudes against latitudes. Because of the
            large number of data points, use black pixel markers (use "k,"
            as the third argument to plt.plot()). Label both axes.
        2. A histogram of the hours of the day, with one bin per hour.
            Label and set the limits of the x-axis.
    """
    plt.clf()
    data = np.load("FARS.npy")  

    hours = data[:, 0]
    longitudes = data[:, 1]
    latitudes = data[:, 2]

    ax1 = plt.subplot(121)
    plt.plot(longitudes, latitudes, 'k,')  
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.axis("equal")  

    ax2 = plt.subplot(122)
    plt.hist(hours, bins=np.arange(25))  
    plt.xlabel("Hour of Day")
    plt.xlim(0, 24)
    
    #plt.subplots_adjust(wspace=0.3)
    #plt.savefig("prob5_plot.png")
    #print("Plot saved to 'prob5_plot.png'")

    plt.show()
        
# Problem 6
def prob6():
    """ Plot the function g(x,y) = sin(x)sin(y)/xy on the domain
    [-2pi, 2pi]x[-2pi, 2pi].
        1. Create 2 subplots: one with a heat map of g, and one with a contour
            map of g. Choose an appropriate number of level curves, or specify
            the curves yourself.
        2. Set the limits of each subplot to [-2pi, 2pi]x[-2pi, 2pi].
        3. Choose a non-default color scheme.
        4. Include a color scale bar for each subplot.
    """
    plt.clf()
    x = np.linspace(-2 * np.pi, 2 * np.pi, 400)
    y = x.copy()
    X, Y = np.meshgrid(x, y)

    numerator = np.sin(X) * np.sin(Y)
    denominator = X * Y
    G = np.where(denominator != 0, numerator / denominator, 1.0)

    plt.subplot(121)
    plt.pcolormesh(X, Y, G, cmap="magma", shading="auto")
    plt.colorbar()
    plt.xlim(-2*np.pi, 2*np.pi)
    plt.ylim(-2*np.pi, 2*np.pi)

    plt.subplot(122)
    plt.contour(X, Y, G, levels=20, cmap="coolwarm") 
    plt.colorbar()
    plt.xlim(-2*np.pi, 2*np.pi)
    plt.ylim(-2*np.pi, 2*np.pi)
    
    #plt.subplots_adjust(wspace=0.3)
    #plt.savefig("prob6_plot.png")
    #print("Plot saved to 'prob6_plot.png'")

    plt.show()

    
if __name__=="__main__":
    print("Hello, world!")

    print("\nTesting Problem 1: Plotting variance of row means")
    prob1()
    
    print("\nTesting Problem 2: Plotting sin(x), cos(x), and arctan(x)")
    prob2()  
    
    print("\nTesting Problem 3: Plotting f(x) = 1 / (x - 1)")
    prob3()
    
    print("\nTesting Problem 4: Plotting sine function variations in subplots")
    prob4()
    
    print("\nTesting Problem 5: Visualizing FARS data with scatter plot and histogram")
    prob5()

    print("\nTesting Problem 6: Heat map and contour plot of g(x, y) = sin(x)sin(y)/xy")
    prob6()

