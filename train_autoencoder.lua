require 'torch'
require 'nn'
require 'nngraph'
require 'Rectifier'
require 'cutorch'
require 'cunn'


local ninput = 4
local noutput = 32
local filterSize = 8
local filterStride = 4


params = {
    game = "ms_pacman",
    dataFile = "../stored_frames/frames_ms_pacman_encoder_0419.t7",
    filename = "ms_pacman" .. "_" .. os.date("%m%d"),
    maxIter = 200,
    learning_rate = 0.01,
    epochs = 1000
}


net = nn.Sequential()
-- encoder
encoder = nn.Sequential()
encoder:add(nn.SpatialConvolution(ninput,noutput, filterSize,filterSize, filterStride,filterStride))
encoder:add(nn.Rectifier())   -- ReLU
net:add(encoder)

-- decoder
decoder = nn.Sequential()
decoder:add(nn.SpatialFullConvolution(noutput, ninput, filterSize, filterSize, filterStride,filterStride))
decoder:add(nn.Rectifier())
net:add(decoder)

net:cuda()

print (net)

--[[
data = torch.rand(4,20,20)
print (data)
print (net:forward(data):size())
print (data)
]]

data = torch.load(params.dataFile).frames:cuda()

dataset={};
function dataset:size() return data:size(1) end -- 100 examples
for i=1,dataset:size() do
    dataset[i] = {data[1], data[1]}
end


criterion = nn.MSECriterion():cuda()
trainer = nn.StochasticGradient(net, criterion)

trainer.learningRate = params.learning_rate
trainer.maxIteration = params.maxIter


for t=1, params.epochs do
  print('Epoch ' .. t, "\t", os.date("%x %X"))

  trainer:train(dataset)

  torch.save("../trained_networks/autoencoders/" .. params.filename .. ".t7", { model = net } )

  print ("Network saved", "\t", os.date("%x %X"))
end
