require 'torch'
require 'nn'
require 'cudnn'
require 'nngraph'

local sft_gan = torch.load('./models/SFT-GAN.t7')
local condition_net = sft_gan['condition_net']
local sr_net = sft_gan['sr_net']
cudnn.convert(condition_net, nn)
cudnn.convert(sr_net, nn)
-- to CPU
condition_net = condition_net:float()
sr_net = sr_net:float()

layers_cond = {}
for i,node in ipairs(condition_net.forwardnodes) do
	local m = node.data.module -- nngraph module
    if m then -- cannot support nngraph containing nngraph
        layers_cond[node.data.forwardNodeId] = m
        print(node.data.forwardNodeId, m)

	end
end
torch.save('./models/condition_net_layers.t7', layers_cond)

layers_sr = {}
for i,node in ipairs(sr_net.forwardnodes) do
	local m = node.data.module -- nngraph module
    if m then -- cannot support nngraph containing nngraph
        layers_sr[node.data.forwardNodeId] = m
        print(node.data.forwardNodeId, m)

	end
end
torch.save('./models/sft_net_layers.t7', layers_sr)
