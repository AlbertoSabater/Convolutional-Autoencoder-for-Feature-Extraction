package.path = package.path .. ";../dqn/?.lua"

require 'torch'
require 'nn'
require 'nngraph'
require 'Rectifier'
require 'cutorch'
require 'cunn'
require 'unsup'


local ninput = 4
local noutput = 32
local filterSize = 8
local filterStride = 4


params = {
    game = "ms_pacman",
    dataFile = "../stored_frames/frames_ms_pacman_0419.t7",
    filename = "ms_pacman" .. "_encoder" .. "_" .. os.date("%m%d"),
    learning_rate = 0.01,
    maxIter = 40,
    epochs = 20,
    numLayers = 1,
    feautureRelation = 2,
    normalizeData = 0,
    gpu = 1,
    convNet = 'convnet_paper1_bigger',
    midLayer = 'none',
    dropout = 0
}

cmd = torch.CmdLine()
cmd:option('-game', params.game, 'game')
cmd:option('-dataFile', params.dataFile, 'path to data')
cmd:option('-filename', params.filename, 'destination path')
cmd:option('-learning_rate', params.learning_rate, 'learning rate')
cmd:option('-maxIter', params.maxIter, 'number of max iterations')
cmd:option('-epochs', params.epochs, 'epochs')
cmd:option('-numLayers', params.numLayers, 'number of layers to train')
cmd:option('-feautureRelation', params.feautureRelation, 'relation to the defined netwokr architecture')
cmd:option('-normalizeData', params.normalizeData, 'true to normalize data')
cmd:option('-gpu', params.gpu, '-1 to use CPU')
cmd:option('-convNet', params.convNet, 'network architecture')
cmd:option('-midLayer', params.midLayer, 'network architecture')
cmd:option('-dropout', params.dropout, 'add a dropout layer before decoder')
options = cmd:parse(arg)

options.filename = "ms_pacman" .. "_encoder"
if options.normalizeData == 1 then
    options.filename = options.filename .. "_norm"
end
if options.dropout == 1 then
    options.filename = options.filename .. "_drop"
elseif options.dropout == 2 then
    options.filename = options.filename .. "_sDrop"
end
options.filename = options.filename .. "_" .. options.numLayers .. "_" .. options.midLayer .. "_" .. os.date("%m%d")

print (options.filename)

for k, v in pairs(options) do
    print(k, v)
end

print ("================================================================")


local msg, cnv = pcall(require, options.convNet)
--cnv = require arg.convNet
args = {}
args.hist_len = 4
args.ncols = 1
args = cnv(args)


net = nn.Sequential()
-- encoder
encoder = nn.Sequential()
-- decoder
decoder = nn.Sequential()




for i=1,math.min(options.numLayers,#args.n_units) do
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

if options.midLayer == "sigmoid" then
    net:add(nn.Sigmoid())
elseif options.midLayer == "tanh" then
    net:add (nn.Tanh())
elseif options.midLayer == "diag" then
    outputFeatures = encoder:get(encoder:size()-1).nOutputPlane
    net:add(nn.Diag(outputFeatures))
end

if options.dropout == 1 then
    net:add(nn.Dropout())
elseif options.dropout == 2 then
    net:add(nn.SpatialDropout())
end
net:add(decoder)

if options.gpu >= 0 then
    net:cuda()
else
    net:double()
end

print (net)

data = torch.load(options.dataFile).frames
if options.gpu >= 0 then
    data = data:cuda()
else
    data = data:double()
end


if options.normalizeData == 1 then
    local mean = {}
    local std = {}
    for i=1,data:size(1) do
         mean[i] = data[i]:mean()
         std[i] = data[i]:std()
         data[i]:add(-mean[i])
         data[i]:div(std[i])
    end
print ("Data normalized")

--[[ Check if datais properly normalized
    for i=1,data:size(1) do
         local testMean = data[i]:mean()
         local testStd = data[i]:std()

         print('       test data, mean:                   ' .. testMean)
         print('       test data, standard deviation:     ' .. testStd)
    end
  ]]
end

dataset={};
function dataset:size() return data:size(1) end -- 100 examples
for i=1,dataset:size() do
    dataset[i] = {data[i], data[i]}
end


criterion = nn.MSECriterion()
if options.gpu >= 0 then
    criterion:cuda()
else
    criterion:double()
end

--local msg, err = pcall(require, 'CustomStochasticGradient', net, criterion)
--print (msg, err)

trainer = require "CustomStochasticGradient"
trainer:init(net, criterion)
--trainer.__init(net, criterion)

trainer.learningRate = options.learning_rate
trainer.maxIteration = options.maxIter


nClock = os.clock()

for t=1, options.epochs do
  currenError = trainer:train(dataset)
end

net:evaluate()
print ("Elapsed time is: ", os.clock()-nClock)
torch.save("../stored_kernels/autoencoders/" .. options.filename .. ".t7", { network = net:get(1), model = net } )
