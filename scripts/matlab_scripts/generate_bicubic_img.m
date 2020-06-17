function generate_bicubic_img()
%% matlab code to genetate mod images, bicubic-downsampled images and bicubic_upsampled images

%% set parameters
% comment the unnecessary line
input_folder = '../../datasets/val_set14/Set14';
save_mod_folder = '../../datasets/val_set14/Set14_mod12';
save_lr_folder = '../../datasets/val_set14/Set14_bicLRx2';
% save_bic_folder = '';

mod_scale = 12;
up_scale = 2;

if exist('save_mod_folder', 'var')
    if exist(save_mod_folder, 'dir')
        disp(['It will cover ', save_mod_folder]);
    else
        mkdir(save_mod_folder);
    end
end
if exist('save_lr_folder', 'var')
    if exist(save_lr_folder, 'dir')
        disp(['It will cover ', save_lr_folder]);
    else
        mkdir(save_lr_folder);
    end
end
if exist('save_bic_folder', 'var')
    if exist(save_bic_folder, 'dir')
        disp(['It will cover ', save_bic_folder]);
    else
        mkdir(save_bic_folder);
    end
end

idx = 0;
filepaths = dir(fullfile(input_folder,'*.*'));
for i = 1 : length(filepaths)
    [paths,imname,ext] = fileparts(filepaths(i).name);
    if isempty(imname)
        disp('Ignore . folder.');
    elseif strcmp(imname, '.')
        disp('Ignore .. folder.');
    else
        idx = idx + 1;
        str_rlt = sprintf('%d\t%s.\n', idx, imname);
        fprintf(str_rlt);

        % read image
        img = imread(fullfile(input_folder, [imname, ext]));
        img = im2double(img);

        % modcrop
        img = modcrop(img, mod_scale);
        if exist('save_mod_folder', 'var')
            imwrite(img, fullfile(save_mod_folder, [imname, '.png']));
        end

        % LR
        im_lr = imresize(img, 1/up_scale, 'bicubic');
        if exist('save_lr_folder', 'var')
            imwrite(im_lr, fullfile(save_lr_folder, [imname, '.png']));
        end

        % Bicubic
        if exist('save_bic_folder', 'var')
            im_bicubic = imresize(im_lr, up_scale, 'bicubic');
            imwrite(im_bicubic, fullfile(save_bic_folder, [imname, '.png']));
        end
    end
end
end

%% modcrop
function img = modcrop(img, modulo)
if size(img,3) == 1
    sz = size(img);
    sz = sz - mod(sz, modulo);
    img = img(1:sz(1), 1:sz(2));
else
    tmpsz = size(img);
    sz = tmpsz(1:2);
    sz = sz - mod(sz, modulo);
    img = img(1:sz(1), 1:sz(2),:);
end
end
