import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data() :
    df = pd.read_csv('t_data.csv')
    print("Features=",df.columns)
    print("Num Years=",len(df)//12,"Num Months=",len(df))
    df['month'] = df.index
    return df[0:150*12],df[150*12:]


def plot_fit(X,y,b,name) :
    plt.plot(X.t,y,label='Actual')
    plt.plot(X.t,X@b,label='Predicted')
    plt.xlabel('Month')
    plt.ylabel('Max Temperature (C)')
    plt.title('Max Temperature - '+name)
    plt.legend()
    plt.savefig('%s_fit.pdf'%name,bbox_inches='tight')
    plt.close()
    
    
def main() :    

    train,test = load_data()

    
if __name__ == "__main__" :
    main()
