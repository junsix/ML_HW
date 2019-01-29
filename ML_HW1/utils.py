import matplotlib.pyplot as plt

def plotData(data):
    fig, ax = plt.subplots(figsize=(20,10))
    results_accepted = data[data.accepted == 1]
    results_rejected = data[data.accepted == 0]
    ax.scatter(results_accepted.test1, results_accepted.test2, marker='+', c='b', s=40)
    ax.scatter(results_rejected.test1, results_rejected.test2, marker='o', c='r', s=30)
    return ax