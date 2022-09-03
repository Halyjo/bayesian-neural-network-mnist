from os import stat
import pyro
from pyro.distributions import Normal, Categorical
from pyro.infer import SVI, Trace_ELBO
import torch
import torch.nn as nn
from train_cnn import LeNet, get_data_loaders
import numpy as np
import matplotlib.pyplot as plt

CLASSES = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')


class BChildLeNet(pyro.nn.PyroModule, LeNet):
    def __init__(self, state_dict_path="lenet_state_dict.pt"):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.sigma = 0.000001

        ## load from state dict
        self.load(state_dict_path)
    
    def gen_prior_dict(self):
        priors = {}
        for key, value in self.state_dict().items():
            priors[key] = Normal(
                loc=value, 
                scale=torch.ones_like(value) * self.sigma,
            )
        return priors


    # def model(x_data, y_data):
    #     # lift module parameters to random variables sampled from the priors
    #     lifted_module = pyro.random_module("module", self, priors)
    #     # sample a regressor (which also samples w and b)
    #     lifted_reg_model = lifted_module()
        
    #     lhat = self.log_softmax(lifted_reg_model(x_data))
        
    #     pyro.sample("obs", Categorical(logits=lhat), obs=y_data)
    #     return lifted_reg_model


def guide(x_data, y_data):
    softplus = torch.nn.Softplus()
    priors = {}

    for key, value in NET.state_dict().items():
        mu = value
        sig = torch.randn_like(value) * SPREAD
        mu_param = pyro.param(key + "_mu", mu)
        sig_param = softplus(pyro.param(key + "_sig", sig))
        priors[key] = Normal(loc=mu_param, scale=sig_param)
        
    # lift module parameters to random variables sampled from the priors
    lifted_module = pyro.random_module("module", NET, priors)
    # sample a regressor (which also samples w and b)
    lifted_reg_model = lifted_module()
    return lifted_reg_model


#######################################################
optimizer = pyro.optim.Adam({"lr": 0.01})
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

num_iterations = 5
loss = 0

train_loader, test_loader = get_data_loaders(batch_size=3)


### Fitting ####
# for j in range(num_iterations):
#     loss = 0
#     for batch_id, data in enumerate(train_loader):
#         # calculate the loss and take a gradient step
#         loss += svi.step(data[0], data[1])
#         # loss += svi.step(data[0].view(-1,28*28), data[1])
#     normalizer_train = len(train_loader.dataset)
#     total_epoch_loss_train = loss / normalizer_train
    
#     print("Epoch ", j, " Loss ", total_epoch_loss_train)
#########################################################

breakpoint()

num_samples = 10

def predict(x):
    sampled_models = [guide(None, None) for _ in range(num_samples)]
    yhats = [model(x).data for model in sampled_models]
    mean = torch.mean(torch.stack(yhats), 0)
    return torch.tensor(np.argmax(mean.numpy(), axis=1))


print('Prediction when network is forced to predict')

correct = 0
total = 0

for j, data in enumerate(test_loader):
    images, labels = data
    breakpoint()
    predicted = predict(images) #.view(-1,28*28))
    correct += (predicted == labels).sum().item()
    total += np.prod(predicted.shape)
print("accuracy: %d %%" % (100 * correct / total))

breakpoint()
##########################################################

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    #plt.imshow(npimg,  cmap='gray')
    #fig.show(figsize=(1,1))
    
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.imshow(npimg,  cmap='gray', interpolation='nearest')
    plt.show()


num_samples = 100
def give_uncertainities(x):
    sampled_models = [guide(None, None) for _ in range(num_samples)]
    yhats = [F.log_softmax(model(x).data, 1).detach().numpy() for model in sampled_models] # x.view(-1,28*28)
    return np.asarray(yhats)


if __name__ == '__main__':    
    pass
