from torchscan import summary
from model.FDADN import FDADN


rfdn = FDADN(upscale=4)



summary(rfdn, (3,960,540))