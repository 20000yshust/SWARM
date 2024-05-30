import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
import matplotlib.pyplot as plt
import cv2
import scipy
import scipy.stats
import torch


class STRIP():
    def __init__(self,  holdout_x, model,device):
        self.holdout_x = holdout_x
        self.x_min = torch.min(self.holdout_x)
        self.x_max = torch.max(self.holdout_x)
        self.model = model
        self.N = 8
        self.detection_boundary = None
        self.device=device

    
    def superimpose(self, X):
        selected_samples = self.holdout_x[random.sample(range(self.holdout_x.shape[0]),self.N),::]
        X=X.cpu().detach().numpy()
        X = np.expand_dims(X,axis=0)
        X = np.vstack([X]*int(selected_samples.shape[0]))
        return torch.from_numpy(np.clip(cv2.addWeighted(selected_samples.cpu().detach().numpy(),1,X,1,0),self.x_min.cpu().detach().numpy(),self.x_max.cpu().detach().numpy())).to(self.device)


    def shannon_entroy(self,y):
        return torch.mean((-1)*torch.nansum(torch.log2(y)*y,dim=1))
    
    
    def determine_detection_boundary(self, FRR=0.01):
        H = []
        trials = 100
        for j in range(trials):
            print(j,end='\r')
            print("[","{0:.2f}".format(100*j/trials),"% complete ] determining the detection boundary....", end='\r')
            x = self.holdout_x[random.choice(range(self.holdout_x.shape[0])),::]
            perturbed_x = self.superimpose(X=x)
            y = self.model(perturbed_x)
            res=self.shannon_entroy(y).cpu().detach().numpy()
            H.append(res)

        (mu, sigma) = scipy.stats.norm.fit(np.array(H))

        self.detection_boundary = scipy.stats.norm.ppf(FRR, loc = mu, scale =  sigma)
        print("\ndetection boundary is ","{0:.5f}".format(self.detection_boundary))
        return H
    
        
    # def draw_perturbed_inputs(self, perturbed_inputs):
    #     plt.cla()
    #     plt.figure(figsize=(25,25))
    #     for i in range(25):
    #         ax = plt.subplot(5,5,i+1)
    #         ax.imshow(perturbed_inputs[i,::])
    #         ax.set_xticks([])
    #         ax.set_yticks([])
    #     plt.tight_layout()
    #     plt.savefig("perturbed_inputs.png")
    #     plt.close()

    
    def detect_backdoor(self,sample_under_test):

        if self.detection_boundary == None:
            self.determine_detection_boundary()
        
        perturbed_samples = self.superimpose(X=sample_under_test)
        # self.draw_perturbed_inputs(perturbed_samples)
        y = self.model(perturbed_samples)
        h = self.shannon_entroy(y)
        print("perturbed sample's shannon entropy is ", "{0:.5f}".format(h))

        if h < self.detection_boundary:
            return 1, h
        else:
            return 0, h