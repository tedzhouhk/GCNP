import numpy as np
import matplotlib.pyplot as plt
import time

def lasso_plot(beta,weight,beta_loss,weight_loss,prune_params,title):
    fig,axs=plt.subplots(2,2)
    fig.suptitle(title)
    fig.subplots_adjust(top=0.8)

    # beta loss
    beta_loss=np.array(beta_loss)
    axs[0,0].plot(np.arange(beta_loss.shape[0]),beta_loss)
    axs[0,0].set_title('Beta Loss')
    axs[0,0].set(xlabel='Epoch',ylabel='Loss')

    # beta
    axs[1,0].hist(beta,bins=int(beta.shape[0]/10))
    axs[1,0].set_title('Beta Distribution')

    # weight loss
    weight_loss=np.array(weight_loss)
    axs[0,1].plot(np.arange(weight_loss.shape[0]),weight_loss)
    axs[0,1].set_title('Weight Loss')
    axs[0,1].set(xlabel='Epoch',ylabel='Loss')

    # parameter
    text=''
    for k,v in prune_params.items():
        text+=str(k)+': '+str(v)+'\n'
    axs[1,1].set_axis_off()
    axs[1,1].text(0,0,text)

    fig.tight_layout()
    fig.savefig('plot/'+time.strftime('%Y%m%d_%H:%M:%S')+'_'+title+'.png')