def LearningWindow(self,x):
        

        if x>0:
            W = self.A_plus*exp(-x/self.tau_plus)

        else:
            W = self.A_minus*exp(x/self.tau_minus)

        return W 