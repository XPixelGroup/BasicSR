clear; close all; clc;

LR_folder = './LR'; % LR
preout_folder = './results'; % pre output
save_folder = './results_20bp';
filepaths  =  dir(fullfile(preout_folder, '*.png'));
max_iter = 20;

if ~ exist(save_folder, 'dir')
    mkdir(save_folder);
end

for idx_im = 1:length(filepaths)
    fprintf([num2str(idx_im) '\n']);
    im_name = filepaths(idx_im).name;
    im_LR = im2double(imread(fullfile(LR_folder, im_name)));
    im_out = im2double(imread(fullfile(preout_folder, im_name)));
    %tic
    im_out = backprojection(im_out, im_LR, max_iter);
    %toc
    imwrite(im_out, fullfile(save_folder, im_name));
end
