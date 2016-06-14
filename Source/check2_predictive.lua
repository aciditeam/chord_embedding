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
nnet=2
step=3

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
-- Define the structure of a model
local structureModel2 = {};
-- Number of layers
structureModel2.nLayers = 1;
-- Size of the input
structureModel2.nInputs =nnet*structureModel.nOutputs;
-- Size of each layer
structureModel2.layers = {structureModel.nOutputs};
-- Number of outputs
structureModel2.nOutputs = noutputs;

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
-- Predictive model definition
-----------------------------
local encoder = defineSimple(structureModel);
local decoder = defineSimple(structureModel2)
-- Turn it into a siamese model (input splits on 1st dimension)
local predictive_encoder = nn.ParallelTable()
-- Add the first part
predictive_encoder:add(encoder)
-- Clone the encoder and share the weight, bias (must also share the gradWeight and gradBias)
for i=2,nnet do
predictive_encoder:add(encoder:clone('weight','bias', 'gradWeight','gradBias'))
end
--The predictive model
local hiddenModel = nn.Sequential()
-- Add the predictive encoder
hiddenModel:add(predictive_encoder)
-- L2 pariwise distance
--model:add(nn.PairwiseDistance(2))
hiddenModel:add(nn.JoinTable(2));
hiddenModel:add(decoder)

local rep_model=nn.Sequential()
rep_model:add(encoder:clone('weight','bias','gradWeight','gradBias'))

local hiddenModel2=nn.ParallelTable()
hiddenModel2:add(hiddenModel)
hiddenModel2:add(rep_model)

local fullModel=nn.Sequential()
fullModel:add(hiddenModel2)
fullModel:add(nn.PairwiseDistance(2))
print(fullModel);


-----------------------------
-- Criterion definition
-----------------------------
--local margin = 1;
fullModel:add(nn.Tanh())
local criterion = nn.MSECriterion();

--------------------------
-- Optimizer definition
--------------------------

optimState = {
   learningRate = 1e-3,
   weightDecay = 0,
   momentum = 0,
   learningRateDecay = 1e-7
}
optimMethod = optim.sgd

if fullModel then
   parameters,gradParameters = fullModel:getParameters()
end

---------------------------
-- Get augmentation
---------------------------

aug_vector={}
i=1
print('===> Here are the data augmentations <===')
for fname,obj in pairs(augmentation) do
    if type(obj) == "function" then
        print(fname)
        aug_vector[i]={obj,fname}
        i=i+1
    end
end

--Probability of augmentations
proportion=0 -- Between 0 and 100 % of augmentation per couple


---------------------------
--Training function
---------------------------
function train(model,trainData,valData,matrix,options)
  -- epoch tracker
  epoch = epoch or 1
  -- time variable
  local time = sys.clock()
  -- adjust the batch size (needs at least 2 examples)
  adjBSize = context --(options.batchSize > 1 and options.batchSize or 2)
  -- set model to training mode (for modules that differ in training and testing, like Dropout)
  fullModel:training();
  -- shuffle order at each epoch
  shuffle = torch.randperm(#trainData);
  -- do one epoch
  print("==> Train epoch # " .. epoch .. ' [batch = ' .. adjBSize .. ' (' .. ((adjBSize * (adjBSize - 1)) / 2) .. ')]')
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
    local inputs1 = torch.Tensor(nnet,mId-t+1,ninputs)
    local inputs2=torch.Tensor(1,mId-t+1,ninputs)
    local k = 1;
    -- iterate over mini-batch examples
    for i = t,(mId - 1) do
      --local precursor=torch.Tensor(nnet,ninputs)

      for l =1,nnet do
        inputs1[l][k]=matrix[trainData[shuffle[i]][l]]
      end
      inputs2[1][k]=matrix[trainData[shuffle[i]][nnet+1]]


      --local to_pred=matrix[trainData[shuffle[i]][nnet+1]]

      --[[rand=math.random(1,100) -- To decide if we make an augmentation
      if rand<proportion then -- If yes
        wh=math.random(1,#aug_vector) -- choose which augmentation
        --print(aug_vector[wh][2])
        if aug_vector[wh][2]=='transpose' then --or aug_vector[wh][2]=='reverse_time' then
          ii1,ii2=aug_vector[wh][1](i1,i2)
          inputs[1][k] = ii1
          inputs[2][k] = ii2
        elseif aug_vector[wh][2]=='reverse_time' then
          inputs[1][k] = i2
          inputs[2][k] = i1
        else
          t=math.random(1,2)  -- Augmentation on i1 or i2?
          if t==1 then
            ii1=aug_vector[wh][1](i1)
            inputs[1][k] = ii1
            inputs[2][k] = i2
          else
            ii2=aug_vector[wh][1](i2)
            inputs[1][k] = i1
            inputs[2][k] = ii2
          end
        end
      else]]
      k = k + 1
    end
    --inputs = torch.Tensor(nnet+1,mId-t+1,ninputs)
    inputs={}
    for dim=1,nnet do
      inputs[dim]=inputs1[dim]
    end
    inputs[nnet+1]={inputs2[1]}
    --inputs=nn.JoinTable(1):forward({inputs1,inputs2})
    --print(inputs:dim())
    --print(inputs:size())

    --if options.type == 'double' then inputs = inputs:double() end
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

        --targets=rep_model:forward(inputs2)
        -- estimate forward pass
        local output = fullModel:forward(inputs)
        -- estimate error (here compare to margin)
        local err = criterion:forward(output[1], output[2])
        -- compute overall error
        f = f + err
        -- estimate df/dW (perform back-prop)
        local df_do = criterion:backward(output[1], output[2])
        fullModel:backward(inputs, df_do)
      -- Normalize gradients and error
      gradParameters:div(inputs:size(2));
      f = f / inputs:size(2)
      -- return f and df/dX
      return f,gradParameters
    end
    -- optimize on current mini-batch
    optimMethod(feval, parameters, optimState)
  end
  -- shuffle order at each epoch
  shuffle2 = torch.randperm(#valData);
  -- do one epoch
  print("==> Val epoch # " .. epoch .. ' [batch = ' .. adjBSize .. ' (' .. ((adjBSize * (adjBSize - 1)) / 2) .. ')]')
  local average_error={}
  average_error[epoch]=0
  for t = 1,#valData,adjBSize do
    -- disp progress
    xlua.progress(t, #valData)
    -- Check size (for last batch)
    bSize = math.min(adjBSize, #valData - t + 1)
    -- Real batch size is combinatorial
    combBSize = ((bSize * (bSize - 1)) / 2)
    -- Maximum indice to account
    mId = math.min(t+context-1,#valData)
    -- create mini batch
    local inputs1 = torch.Tensor(nnet,mId-t+1,ninputs)
    local inputs2=torch.Tensor(1,mId-t+1,ninputs)
    local k = 1;
    -- iterate over mini-batch examples
    for i = t,(mId - 1) do

      --local precursor=torch.Tensor(nnet,ninputs)

      for l =1,nnet do
        inputs[l][k]=matrix[valData[shuffle2[i]][l]]
      end
      inputs2[k]=matrix[valData[shuffle2[i]][nnet+1]]

      k = k + 1
    end
    inputs = torch.Tensor(nnet+1,mId-t+1,ninputs)
    inputs=nn.JoinTable(1):forward({inputs1,inputs2})
    local f=0
    -- estimate forward pass
    --local targets = rep_model:forward(inputs2)
    local output = model:forward(inputs)
    -- estimate error (here compare to margin)
    local err = criterion:forward(output[1], output[2])
    -- compute overall error
    f = f + err
    average_error[epoch]=average_error[epoch]+err/(k-1)
  end
    -- time taken
    time = sys.clock() - time;
    time = time / (#trainData+#valData);
    print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')
    -- next epoch
    epoch = epoch + 1
end
-------------------------------
-- Creating inputs to NN
-------------------------------

massive_matrix=torch.load("final_data/massive_matrix.dat",'ascii')
time_pointer=torch.load("final_data/time_pointer.dat",'ascii')
pitch_per_inst=torch.load('final_data/pitch_per_inst.dat','ascii')
--total_track=torch.load('final_data/total_track.dat','ascii')
--track_list=torch.load('final_data/track_list.dat',ascii)
--track_labels_total=torch.load('final_data/track_labels_total.dat','ascii')
event_total=torch.load('final_data/event_total.dat','ascii')

--  10%
local piece_num=9
local size=time_pointer[piece_num]
--time_pointer[9]=7683

--------------------------------
-- Data description
--------------------------------
-- Massive_matrix is concatenation of all the pieces and all instruments
-- event_total (0 or 1) indicates when are the events happening in massive_matrix
-- track_list[i] is the name of the i-th track in massive_matrix
-- Piece i begins at time_pointer[i-1]+1 and finishes at time_pointer[i]
-- total_track[i] is the name of the i-th instrument
-- The i-th instrument plays from pitch_per_inst[1] to pitch_per_inst[2]
-- track_labels_total[i] is the list of instrument in the i-th piece

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
--]]--


load_model=false
if load_model==false then
  -- Determine couples train/validation
  trainData={}
  valData={}
  j=1
  l=1
  for i=1,#event_pointer do
    for k=1,#event_pointer[i]-1-nnet,step do
      if event_pointer[i][k]+nnet+1<size then  -- Validation set
        valData[j]={}
        for h=1,nnet+1 do
          valData[j][h]=event_pointer[i][k+h-1]
          if valData[j][h]==nil then print('a') end
        end
        j=j+1
      else  -- Training set
        trainData[l]={}
        for h=1,nnet+1 do
          trainData[l][h]=event_pointer[i][k+h-1]
          if trainData[l][h]==nil then print('a') end
        end
        l=l+1
      end
    end
  end

--print(j)
--print(l)
--print(#trainData)

  print ('===> training <===')
  -- Do the training
  epoch=1
  while epoch<30 do
    train(fullModel,trainData,valData,massive_matrix:t(),options)
  end
  -- Save net and mean error on validation
  filename='Models/model_predictive'..tostring(proportion)..'.net'
  filename2='Results/average_error_predictive'..tostring(proportion)..'dat'
  torch.save(filename,fullModel)
  torch.save(filename2,average_error,'ascii')

elseif load_model==true then
  -- Load net
  filename='Models/model_predictive'..tostring(proportion)..'.net'
  fullModel=torch.load(filename)
  print('=====> Embedding calculation <=====')
-- if encoder doesn't work, save encoder in the first place...
-- Calculate embedding representations
output_rep=torch.Tensor(noutputs,massive_matrix:size(2))
for i=1,massive_matrix:size(2) do
  output_rep:t()[i]=encoder:forward(massive_matrix:t()[i])
end

adjBSize = context
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
    uv=uv+1  -- To check the final size of outputs_rep
  end
  --if k-1 ~= 64 then print(k-1) end
  outputs=encoder:forward(inputs)
  output_rep:t()[{{t,(mId-1)}}]=outputs
end
-- Export embedding
filename3='Results/output_predictive'..tostring(proportion)..'.dat'
torch.save(filename3,output_rep,'ascii')

  print(massive_matrix:size(2))
  print('===>Checking output_rep')

  for i=1,output_rep:size(2) do
    if output_rep:t()[i][1]==nil then print(i) end
  end
end
