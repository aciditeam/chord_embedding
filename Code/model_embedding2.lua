require 'torch'
require 'nn'
local nninit = require 'nninit'
require 'optim'
require 'xlua'
require 'augmentation'

math.randomseed(os.time())

--Pitch_range : 2485
--Length : 79422
use_cuda = false
context=64 -- Quarter note /Batch size
ninputs=2485 -- Pitch_range
noutputs=50

-----------------------------
-- Model definition functions
-----------------------------
-- Define the structure of a model
local structureModel = {};
-- Number of layers
structureModel.nLayers = 5;
-- Size of the input
structureModel.nInputs = ninputs;
-- Size of each layer
structureModel.layers = {1000, 1000, 500, 200, 50};
-- Number of outputs
structureModel.nOutputs = noutputs;
-- Define a single network
function defineSimple(structure)
  -- Handle the use of CUDA
  if use_cuda then local nn = require 'cunn' else local nn = require 'nn' end
  -- Container
  local model = nn.Sequential();
  -- Hidden layers
  for i = 1,structure.nLayers do
    -- Linear transform
    if i == 1 then
      model:add(nn.Linear(structure.nInputs,structure.layers[i]));
    else
      model:add(nn.Linear(structure.layers[i-1],structure.layers[i]));
    end
    -- Batch normalization
    model:add(nn.BatchNormalization(structure.layers[i]));
    -- Non-linearity
    model:add(nn.ReLU());
    -- Dropout
    model:add(nn.Dropout(0.5));
  end
  -- Final regression layer
  model:add(nn.Linear(structure.layers[structure.nLayers],structure.nOutputs))
  -- Initialize the weights of the network by finding only the linear modules
  linearNodes = model:findModules('nn.Linear')
  -- Initialize weights
  for l = 1,#linearNodes do
    module = linearNodes[l];
    module:init('weight', nninit.xavier);
    module:init('bias', nninit.xavier);
  end
  return model
end

-----------------------------
-- Siamese model definition
-----------------------------
local encoder = defineSimple(structureModel);
-- Turn it into a siamese model (input splits on 1st dimension)
local siamese_encoder = nn.ParallelTable()
-- Add the first part
siamese_encoder:add(encoder)
-- Clone the encoder and share the weight, bias (must also share the gradWeight and gradBias)
siamese_encoder:add(encoder:clone('weight','bias', 'gradWeight','gradBias'))
--The siamese model
local model = nn.Sequential()
-- Add the siamese encoder
model:add(siamese_encoder)
-- L2 pariwise distance
model:add(nn.PairwiseDistance(2))
--model:add(nn.JoinTable());
print(model);

-----------------------------
-- Criterion definition
-----------------------------
local margin = 1;
local criterion = nn.HingeEmbeddingCriterion(margin);
-- Other potential choices
-- nn.MarginRankingCriterion(margin)
-- nn.TripletEmbeddingCriterion

--------------------------
-- Optimizer definition
--------------------------
-- For sgd
optimState = {
   learningRate = 1e-3,
   weightDecay = 0,
   momentum = 0,
   learningRateDecay = 1e-7
}
optimMethod = optim.sgd

if model then
   parameters,gradParameters = model:getParameters()
end

---------------------------
--Training function
---------------------------

function train(model,trainData,matrix,options)
  -- epoch tracker
  epoch = epoch or 1
  -- time variable
  local time = sys.clock()
  -- adjust the batch size (needs at least 2 examples)
  adjBSize = context --(options.batchSize > 1 and options.batchSize or 2)
  -- set model to training mode (for modules that differ in training and testing, like Dropout)
  model:training();
  -- shuffle order at each epoch
  shuffle = torch.randperm(#trainData);
  -- do one epoch
  print("==> epoch # " .. epoch .. ' [batch = ' .. adjBSize .. ' (' .. ((adjBSize * (adjBSize - 1)) / 2) .. ')]')
  for t = 1,#trainData,adjBSize do
    -- disp progress
    xlua.progress(t, #trainData)
    -- Check size (for last batch)
    bSize = math.min(adjBSize, #trainData - t + 1)
    -- Real batch size is combinatorial
    combBSize = ((bSize * (bSize - 1)) / 2)
    -- Maximum indice to account
    mId = math.min(t+context-1,#trainData)
    -- create mini batch
    local inputs = {}
    local targets = torch.Tensor(mId-t);
    local k = 1;
    inputs[1] = torch.Tensor(mId-t, ninputs);
    inputs[2] = torch.Tensor(mId-t, ninputs);
    -- iterate over mini-batch examples
    for i = t,(mId - 1) do
      -- load first sample
      local i1 = matrix[trainData[shuffle[i]][1]]
      -- load second sample
      local i2 = matrix[trainData[shuffle[i]][2]]
      -- load label
      local label=trainData[shuffle[i]][3]

      inputs[1][k] = i1
      inputs[2][k] = i2
      targets[k] = label;
      k = k + 1

    end
    -- create closure to evaluate f(X) and df/dX
    local feval = function(x)
      -- get new parameters
      if x ~= parameters then
        parameters:copy(x)
      end
      -- reset gradients
      gradParameters:zero()
      -- f is the average of all criterions
      local f = 0

      -- [[ Evaluate function for each example of the mini-batch ]]--
--[[     ---------If we cannot use matrix calculation straight forward-----
      for i = 1,#inputs do
        -- estimate forward pass
        local output = model:forward(inputs[i])
        -- estimate error (here compare to margin)
        local err = criterion:forward(output, targets[i])
        -- compute overall error
        f = f + err
        -- estimate df/dW (perform back-prop)
        local df_do = criterion:backward(output, targets[i])
        model:backward(inputs[i], df_do)
      end
      --]]

        -- estimate forward pass
        local output = model:forward(inputs)
        -- estimate error (here compare to margin)
        local err = criterion:forward(output, targets)
        -- compute overall error
        f = f + err
        -- estimate df/dW (perform back-prop)
        local df_do = criterion:backward(output, targets)
        model:backward(inputs, df_do)
      -- Normalize gradients and error
      gradParameters:div(#inputs);
      f = f / #inputs
      -- return f and df/dX
      return f,gradParameters
    end
    -- optimize on current mini-batch
    optimMethod(feval, parameters, optimState)
  end
  -- time taken
  time = sys.clock() - time;
  print("\n==> time to learn 1 epoch = " .. (time/60) .. 'min')
  time = time / #trainData;
  print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

  epoch = epoch + 1
end



--Massive_matrix is concatenation of all the pieces and all instruments
-- event_total (0 or 1) indicates when are the events happening in massive_matrix

massive_matrix=torch.load("final_data/massive_matrix.csv",'ascii')
time_pointer=torch.load("final_data/time_pointer.csv",'ascii')
--pitch_per_inst=torch.load('final_data/pitch_per_inst.csv','ascii')
--track_labels_total=torch.load('final_data/track_labels_total.csv','ascii')
event_total=torch.load('final_data/event_total.csv','ascii')

--------------------------------
-- Data description
--------------------------------
-- Piece i begins at time_pointer[i-1]+1 and finishes at time_pointer[i]
-- track_labels_total[i] is the name of the i-th instrument
-- The i-th instrument plays from pitch_per_inst[1] to pitch_per_inst[2]


------- Some functions to define-------
function sum(t)
    local sum = 0
    for k,v in pairs(t) do
        sum = sum + v
    end
    return sum
end

local function round(x)
return math.floor(x+0.5)
end

-------------------Just to know and calculate a proportion----------
--local n_events=sum(event_total)


j=1
h=1
event_pointer={}
event_pointer[j]={}
--Make a list of times of event depending on the piece
for i=1,#event_total do
  -- If next event is on the next piece
  if i==time_pointer[j]+1 then
    j=j+1
    event_pointer[j]={}
    h=1
  end
  --If there is an event
  if event_total[i]==1 then
    event_pointer[j][h]=i
    h=h+1
  end
end

-------- Check -------
--[[
event_length=0
for i=1,#event_pointer do
event_length=event_length+#event_pointer[i]
end
print(event_length)
print(sum(event_total))
]]--


load_model=true

if load_model==false then
  -- Determine couples
  couples_labels={}
  j=1
  for i=1,#event_pointer do
    for k=1,#event_pointer[i]-1 do
      --Picking similar events
      couples_labels[j]={event_pointer[i][k],event_pointer[i][k+1],1}
      j=j+1
      if couples_labels[j-1][1]==nil then print('a') end

      --Picking dissimilar events
      ::tryagain::
      h=math.random(1,#event_pointer)
      if h==i then goto tryagain end
      w=math.random(1,#event_pointer[h])
      -- Writing the two vectors coordinate and the label
      couples_labels[j]={event_pointer[i][k],event_pointer[h][w],-1}
      j=j+1
      if couples_labels[j-1][1]==nil then print('a') end
    end
  end

  print ('===> training <===')
  -- Do the training
  epoch=1
  while epoch<100 do
    train(model,couples_labels,massive_matrix:t(),options)
  end
  -- Save net
  filename='Models/model.net'
  torch.save(filename,model)

elseif load_model==true then
  -- Load net
  filename='Models/model.net'
  model=torch.load(filename)
  print('encoder')
  print(encoder)
  print('=====> Embedding calculation <=====')
-- if encoder doesn't work, save encoder in the first place...
-- Calculate embedding representations
output_rep=torch.Tensor(noutputs,massive_matrix:size(2))
for i=1,massive_matrix:size(2) do
  --output_rep:t()[i]=encoder:forward(massive_matrix:t()[i])
end

adjBSize = context --(options.batchSize > 1 and options.batchSize or 2)
uv=0
for t = 1,massive_matrix:size(2),adjBSize do

  -- Check size (for last batch)
  bSize = math.min(adjBSize,massive_matrix:size(2) - t + 1)
  -- Maximum indice to account
  mId = math.min(t+adjBSize,massive_matrix:size(2)+1)
  --mId = math.min(t+adjBSize-1,massive_matrix:size(2))


  -- create mini batch
  local inputs = torch.Tensor(mId-t, ninputs);
  local k = 1;
  --print(t,mId-1)
  -- iterate over mini-batch examples
  for i = t,(mId - 1) do
    -- load first sample
    local i1 = massive_matrix:t()[i]
    inputs[k] = i1
    k = k + 1
    uv=uv+1
  end
  --if k-1 ~= 64 then print(k-1) end
  outputs=encoder:forward(inputs)
  output_rep:t()[{{t,(mId-1)}}]=outputs
end
-- Export embedding
torch.save('Results/output_siamese1.csv',output_rep,'ascii')
end

print(massive_matrix:size(2))
print('Checking')

for i=1,output_rep:size(2) do
  if output_rep:t()[i][1]==nil then print(i) end
end

--parcours vecteur d'evenement -->  Check
-- output2= encoder:forward(event)
--Save model torch.save(filename,model)
--export embedding
