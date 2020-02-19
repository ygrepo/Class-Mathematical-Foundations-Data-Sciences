
from findata.findata_tools import *
import numpy as np

def qa():
    names, days = load_data('stockprices.csv')
    (nDays, nStocks) = days.shape
    print(days.shape)
    nReturns = nDays - 1
    returns = days[1:] - days[:-1]
    print(returns)

    returnsCentered = returns - np.mean(returns, axis=0)
    cov = np.cov(returnsCentered,rowvar=False)
    U, S, returnsPrincipalDirections = np.linalg.svd(np.cov(returnsCentered,rowvar=False))
    maxCoeffStocks_index = [np.argmax(np.abs(returnsPrincipalDirections[k])) for k in range(2)]
    maxCoeffStocks = [names[k] for k in maxCoeffStocks_index]

    print(maxCoeffStocks)
    pretty_print(np.sum(np.abs(returns),axis = 0),names)


if __name__ == "__main__":
    qa()



