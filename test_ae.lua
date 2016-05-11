require 'torch'
require 'nn'
require 'nngraph'
require 'cutorch'
require 'unsup'



nfiltersin = 1
nfiltersout = 16
kernelsize = 9
inputsize = 25

beta = 1


-- params:
conntable = nn.tables.full(nfiltersin, nfiltersout)
kw, kh = kernelsize, kernelsize
iw, ih = inputsize, inputsize


-- connection table:
local decodertable = conntable:clone()
decodertable[{ {},1 }] = conntable[{ {},2 }]
decodertable[{ {},2 }] = conntable[{ {},1 }]
local outputFeatures = conntable[{ {},2 }]:max()


   -- encoder:
encoder = nn.Sequential()
encoder:add(nn.SpatialConvolutionMap(conntable, kw, kh, 1, 1))
encoder:add(nn.Tanh())
encoder:add(nn.Diag(outputFeatures))

-- decoder:
decoder = nn.Sequential()
decoder:add(nn.SpatialFullConvolutionMap(decodertable, kw, kh, 1, 1))

-- complete model
module = unsup.AutoEncoder(encoder, decoder, beta)


print (encoder)
print (decoder)

print (encoder:get(1).weight:size())
print (decoder:get(1).weight:size())
