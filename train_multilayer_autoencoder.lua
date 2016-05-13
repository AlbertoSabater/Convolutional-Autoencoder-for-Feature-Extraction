package.path = package.path .. ";../dqn/?.lua"

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
    dataFile = "../stored_frames/frames_ms_pacman_0419.t7",
    filename = "ms_pacman" .. "_encoder" .. "_" .. os.date("%m%d"),
    maxIter = 200,
    learning_rate = 0.01,
    epochs = 1000,
    numLayers = 2,
    feautureRelation = 2
}


cnv = require 'convnet_paper1_bigger'
args = {}
args.hist_len = 4
args.ncols = 1
args = cnv(args)


net = nn.Sequential()
-- encoder
encoder = nn.Sequential()
-- decoder
decoder = nn.Sequential()




for i=1,math.min(params.numLayers,#args.n_units) do
    if i == 1 then  -- First convolutional layer
      encoder:add(nn.SpatialConvolution(args.hist_len*args.ncols, args.n_units[i]/2,
                          args.filter_size[i], args.filter_size[i],
                          args.filter_stride[i], args.filter_stride[i],1))

      decoder:add(nn.SpatialFullConvolution(args.n_units[i]/2, args.hist_len*args.ncols,
                          args.filter_size[i], args.filter_size[i],
                          args.filter_stride[i],args.filter_stride[i]))
      decoder:add(args.nl())

    else
      encoder:add(nn.SpatialConvolution(encoder:get((i-1)*2 -1).nOutputPlane, args.n_units[i]/2,
                      args.filter_size[i], args.filter_size[i],
                      args.filter_stride[i], args.filter_stride[i]))

      decoder:insert(args.nl(), 1)
      decoder:insert(nn.SpatialFullConvolution(args.n_units[i]/2, encoder:get((i-1)*2 -1).nOutputPlane,
                          args.filter_size[i], args.filter_size[i],
                          args.filter_stride[i],args.filter_stride[i]), 1)
    end

    encoder:add(args.nl())

end


net:add(encoder)
net:add(decoder)

net:cuda()

print (net)

--[[
data = torch.rand(1,4,84,84):cuda()
print (data:size())
print(net:forward(data):size())
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
