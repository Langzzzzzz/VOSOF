#!/usr/bin/env python

import getopt
import math
import numpy
import PIL
import PIL.Image
import sys
import torch
import torch.nn.functional as F
import cv2
import numpy as np

try:
    from .correlation import correlation # the custom cost volume layer
except:
    sys.path.insert(0, './correlation'); import correlation # you should consider upgrading python
# end

##########################################################

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################

args_strModel = 'default' # 'default', or 'chairs-things'
# args_strOne = './images/one.png'
# args_strTwo = './images/two.png'
# args_strOut = './out.flo'

# for strOption, strArg in getopt.getopt(sys.argv[1:], '', [
#     'model=',
#     'one=',
#     'two=',
#     'out=',
# ])[0]:
#     if strOption == '--model' and strArg != '': args_strModel = strArg # which model to use
#     if strOption == '--one' and strArg != '': args_strOne = strArg # path to the first frame
#     if strOption == '--two' and strArg != '': args_strTwo = strArg # path to the second frame
#     if strOption == '--out' and strArg != '': args_strOut = strArg # path to where the output should be stored
# # end

##########################################################

backwarp_tenGrid = {}
backwarp_tenPartial = {}

def backwarp(tenInput, tenFlow):
    if str(tenFlow.shape) not in backwarp_tenGrid:
        tenHor = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1, 1, 1, -1).repeat(1, 1, tenFlow.shape[2], 1)
        tenVer = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1, -1, 1).repeat(1, 1, 1, tenFlow.shape[3])

        backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([ tenHor, tenVer ], 1).cuda()
    # end

    if str(tenFlow.shape) not in backwarp_tenPartial:
        backwarp_tenPartial[str(tenFlow.shape)] = tenFlow.new_ones([ tenFlow.shape[0], 1, tenFlow.shape[2], tenFlow.shape[3] ])
    # end

    tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] * (2.0 / (tenInput.shape[3] - 1.0)), tenFlow[:, 1:2, :, :] * (2.0 / (tenInput.shape[2] - 1.0)) ], 1)
    tenInput = torch.cat([ tenInput, backwarp_tenPartial[str(tenFlow.shape)] ], 1)

    tenOutput = torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=True)

    tenMask = tenOutput[:, -1:, :, :]; tenMask[tenMask > 0.999] = 1.0; tenMask[tenMask < 1.0] = 0.0

    return tenOutput[:, :-1, :, :] * tenMask
# end

##########################################################

class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()

        class Extractor(torch.nn.Module):
            def __init__(self):
                super().__init__()

                self.netOne = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netTwo = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netThr = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netFou = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netFiv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netSix = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=128, out_channels=196, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )
            # end

            def forward(self, tenInput):
                tenOne = self.netOne(tenInput)
                tenTwo = self.netTwo(tenOne)
                tenThr = self.netThr(tenTwo)
                tenFou = self.netFou(tenThr)
                tenFiv = self.netFiv(tenFou)
                tenSix = self.netSix(tenFiv)

                return [ tenOne, tenTwo, tenThr, tenFou, tenFiv, tenSix ]
            # end
        # end

        class Decoder(torch.nn.Module):
            def __init__(self, intLevel):
                super().__init__()

                intPrevious = [ None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None ][intLevel + 1]
                intCurrent = [ None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None ][intLevel + 0]

                if intLevel < 6: self.netUpflow = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)
                if intLevel < 6: self.netUpfeat = torch.nn.ConvTranspose2d(in_channels=intPrevious + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=4, stride=2, padding=1)
                if intLevel < 6: self.fltBackwarp = [ None, None, None, 5.0, 2.5, 1.25, 0.625, None ][intLevel + 1]

                self.netOne = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netTwo = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netThr = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128 + 128, out_channels=96, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netFou = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netFiv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netSix = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=3, stride=1, padding=1)
                )
            # end

            def forward(self, tenOne, tenTwo, objPrevious):
                tenFlow = None
                tenFeat = None

                if objPrevious is None:
                    tenFlow = None
                    tenFeat = None

                    tenVolume = torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(tenOne=tenOne, tenTwo=tenTwo), negative_slope=0.1, inplace=False)

                    tenFeat = torch.cat([ tenVolume ], 1)

                elif objPrevious is not None:
                    tenFlow = self.netUpflow(objPrevious['tenFlow'])
                    tenFeat = self.netUpfeat(objPrevious['tenFeat'])

                    tenVolume = torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(tenOne=tenOne, tenTwo=backwarp(tenInput=tenTwo, tenFlow=tenFlow * self.fltBackwarp)), negative_slope=0.1, inplace=False)

                    tenFeat = torch.cat([ tenVolume, tenOne, tenFlow, tenFeat ], 1)

                # end

                tenFeat = torch.cat([ self.netOne(tenFeat), tenFeat ], 1)
                tenFeat = torch.cat([ self.netTwo(tenFeat), tenFeat ], 1)
                tenFeat = torch.cat([ self.netThr(tenFeat), tenFeat ], 1)
                tenFeat = torch.cat([ self.netFou(tenFeat), tenFeat ], 1)
                tenFeat = torch.cat([ self.netFiv(tenFeat), tenFeat ], 1)

                tenFlow = self.netSix(tenFeat)

                return {
                    'tenFlow': tenFlow,
                    'tenFeat': tenFeat
                }
            # end
        # end

        class Refiner(torch.nn.Module):
            def __init__(self):
                super().__init__()

                self.netMain = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=81 + 32 + 2 + 2 + 128 + 128 + 96 + 64 + 32, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8, dilation=8),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=16, dilation=16),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1)
                )
            # end

            def forward(self, tenInput):
                return self.netMain(tenInput)
            # end
        # end

        self.netExtractor = Extractor()

        self.netTwo = Decoder(2)
        self.netThr = Decoder(3)
        self.netFou = Decoder(4)
        self.netFiv = Decoder(5)
        self.netSix = Decoder(6)

        self.netRefiner = Refiner()

        self.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.hub.load_state_dict_from_url(url='http://content.sniklaus.com/github/pytorch-pwc/network-' + args_strModel + '.pytorch', file_name='pwc-' + args_strModel).items() })
    # end

    def forward(self, tenOne, tenTwo):
        tenOne = self.netExtractor(tenOne)
        tenTwo = self.netExtractor(tenTwo)

        objEstimate = self.netSix(tenOne[-1], tenTwo[-1], None)
        objEstimate = self.netFiv(tenOne[-2], tenTwo[-2], objEstimate)
        objEstimate = self.netFou(tenOne[-3], tenTwo[-3], objEstimate)
        objEstimate = self.netThr(tenOne[-4], tenTwo[-4], objEstimate)
        objEstimate = self.netTwo(tenOne[-5], tenTwo[-5], objEstimate)

        return (objEstimate['tenFlow'] + self.netRefiner(objEstimate['tenFeat'])) * 20.0
    # end
# end

netNetwork = None

##########################################################

def estimate(tenOne, tenTwo):
    global netNetwork

    if netNetwork is None:
        netNetwork = Network().cuda().eval()
    # end

    assert(tenOne.shape[1] == tenTwo.shape[1])
    assert(tenOne.shape[2] == tenTwo.shape[2])

    intWidth = tenOne.shape[2]
    intHeight = tenOne.shape[1]

    # assert(intWidth == 1024) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
    # assert(intHeight == 436) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

    tenPreprocessedOne = tenOne.cuda().view(1, 3, intHeight, intWidth)
    tenPreprocessedTwo = tenTwo.cuda().view(1, 3, intHeight, intWidth)

    intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
    intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))

    tenPreprocessedOne = torch.nn.functional.interpolate(input=tenPreprocessedOne, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
    tenPreprocessedTwo = torch.nn.functional.interpolate(input=tenPreprocessedTwo, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

    tenFlow = torch.nn.functional.interpolate(input=netNetwork(tenPreprocessedOne, tenPreprocessedTwo), size=(intHeight, intWidth), mode='bilinear', align_corners=False)

    tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
    tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

    return tenFlow[0, :, :, :].cpu()
# end

##########################################################

def calcOpticalFlow(img1, img2):
    # tenOne = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(PIL.Image.open(img1))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
    # tenTwo = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(PIL.Image.open(img2))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
    # print(tenOne.shape)
    tenOutput = estimate(img1, img2)

    # objOutput = open(output, 'wb')

    # numpy.array([ 80, 73, 69, 72 ], numpy.uint8)
    # # 1024 436
    # numpy.array([ tenOutput.shape[2], tenOutput.shape[1] ], numpy.int32)
    # # shape (436, 1024, 2)
    res = numpy.array(tenOutput.numpy().transpose(1, 2, 0), numpy.float32)
    # print(tenOutput.numpy().transpose(1, 2, 0))
    # objOutput.close()
    # numpy.save(output_file, res)

    # data = numpy.load("test_out.npy")
    # print(data)
    return res

def warp_image(image, flow):
    """
    Warps an image with the given flow.
    """
    B, C, H, W = image.size()
    print(str(B) + " " + str(C) + " " + str(H) + " " + str(W))

    # Generate mesh grid.
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    flow = torch.from_numpy(flow).unsqueeze(0).permute(0, 3, 1, 2)

    if image.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flow

    # Normalize to -1,1
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:] / max(W-1,1) - 1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:] / max(H-1,1) - 1.0

    warped_image = F.grid_sample(image, vgrid.permute(0,2,3,1), mode='bilinear', padding_mode='zeros')
    return warped_image

def load_image_as_tensor(img):

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert the NumPy array to a PyTorch tensor and reshape it
    img_tensor = torch.FloatTensor(np.ascontiguousarray(img.transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)))

    # Add a batch dimension if needed (for a single image, make it a batch of 1)
    img_tensor = img_tensor.unsqueeze(0)  # Shape becomes (1, C, H, W)
    return img_tensor

def generate_warped_image(img):
    warped_image_cpu = img.cpu()
    warped_image_cpu = warped_image_cpu * 255
    warped_image_np = warped_image_cpu.numpy().astype('uint8')

    warped_image_np = warped_image_np.squeeze(0).transpose((2, 0, 1))

    return warped_image_np

if __name__ == '__main__':
    # tenOne = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(PIL.Image.open(args_strOne))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
    # tenTwo = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(PIL.Image.open(args_strTwo))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))

    # tenOutput = estimate(tenOne, tenTwo)

    # objOutput = open(args_strOut, 'wb')

    # numpy.array([ 80, 73, 69, 72 ], numpy.uint8).tofile(objOutput)
    # # 1024 436
    # numpy.array([ tenOutput.shape[2], tenOutput.shape[1] ], numpy.int32).tofile(objOutput)
    # # shape (436, 1024, 2)
    # numpy.array(tenOutput.numpy().transpose(1, 2, 0), numpy.float32).tofile(objOutput)
    # objOutput.close()

    im = PIL.Image.open("/root/autodl-tmp/data/DAVIS/2017/trainval/JPEGImages/480p/bear/00000.jpg")

    print(numpy.array(im).shape)
    print(type(numpy.array(im).size))
    img1 = "/root/autodl-tmp/data/DAVIS/2017/trainval/JPEGImages/480p/bear/00000.jpg"
    img2 = "/root/autodl-tmp/data/DAVIS/2017/trainval/JPEGImages/480p/bear/00001.jpg"
    output_file = "test_out.npy"

    res = calcOpticalFlow(img1, img2)
   
    numpy.save(output_file, res)

    data = numpy.load(output_file)
    img2 = numpy.array(PIL.Image.open(img2))
    img2 = load_image_as_tensor(img2)
    warped_image2 = warp_image(img2, data)
    print(warped_image2.size())

    warped_image2 = generate_warped_image(warped_image2)
    warped_image_pil = PIL.Image.fromarray(warped_image2)
    warped_image_pil.save('warped_image.png')

    
    if img2.is_cuda:
        img2 = img2.cpu()
    img2 = img2.numpy()
    img2 = img2.squeeze(0).transpose((2, 0, 1))
    print(type(warped_image2))
    print(warped_image2.shape)
    print(type(img2))
    print(img2.shape)
    difference = np.abs(warped_image2 - img2)
    alpha = 5.0  # Adjust based on your needs for sensitivity
    confidence = np.exp(-alpha * difference)
    confidence = np.clip(confidence, 0, 1)

    print(confidence)
    print(confidence.shape)