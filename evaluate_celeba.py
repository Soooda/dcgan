import torch
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp

from model.dcgan import Generator

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

z = 100
channel = 3
checkpoint = osp.sep.join(("checkpoints", "celeba.pth"))
model = Generator(1, z, channel).to(device)

with torch.no_grad():
    model.eval()
    temp = torch.load(checkpoint, map_location=device)
    ret = model.load_state_dict(temp['netG'])
    print(ret)

    for i in range(10):
        noise = torch.randn(1, z, 1, 1, device=device)
        generated_data = model(noise).cpu()[0].numpy()
        generated_data = np.transpose(generated_data, (1, 2, 0))

        # print("Max: {} Min: {}".format(np.amax(generated_data), np.amin(generated_data)))

        plt.figure(figsize=(4, 4))
        plt.axis('Off')
        plt.imshow((generated_data + 1) / 2)
        plt.show()
