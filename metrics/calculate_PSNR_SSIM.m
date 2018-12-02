function calculate_PSNR_SSIM()

% GT and SR folder
folder_GT = '/mnt/SSD/xtwang/BasicSR_datasets/val_set5/Set5';
folder_SR  = '/home/xtwang/Projects/BasicSR/results/RRDB_PSNR_x4/set5';
scale = 4;
suffix = '';  % suffix for SR images
test_Y = 1;  % 1 for test Y channel only; 0 for test RGB channels
if test_Y
    fprintf('Tesing Y channel.\n');
else
    fprintf('Tesing RGB channels.\n');
end
filepaths = dir(fullfile(folder_GT, '*.png'));
PSNR_all = zeros(1, length(filepaths));
SSIM_all = zeros(1, length(filepaths));

for idx_im = 1:length(filepaths)
    im_name = filepaths(idx_im).name;
    im_GT = imread(fullfile(folder_GT, im_name));
    im_SR  = imread(fullfile(folder_SR, [im_name(1:end-4), suffix, '.png']));

    if test_Y  % evaluate on Y channel in YCbCr color space
        if size(im_GT, 3) == 3
            im_GT_YCbCr = rgb2ycbcr(im2double(im_GT));
            im_GT_in = im_GT_YCbCr(:,:,1);
            im_SR_YCbCr = rgb2ycbcr(im2double(im_SR));
            im_SR_in = im_SR_YCbCr(:,:,1);
        else
            im_GT_in = im2double(im_GT);
            im_SR_in = im2double(im_SR);
        end
    else  % evaluate on RGB channels
        im_GT_in = im2double(im_GT);
        im_SR_in = im2double(im_SR);
    end

    % calculate PSNR and SSIM
    PSNR_all(idx_im) = calculate_PSNR(im_GT_in * 255, im_SR_in * 255, scale);
    SSIM_all(idx_im) = calculate_SSIM(im_GT_in * 255, im_SR_in * 255, scale);
    fprintf('%d.(X%d)%20s: \tPSNR = %f \tSSIM = %f\n', idx_im, scale, im_name(1:end-4), PSNR_all(idx_im), SSIM_all(idx_im));
end

fprintf('\n%26s: \tPSNR = %f \tSSIM = %f\n', '####Average', mean(PSNR_all), mean(SSIM_all));
end

function res = calculate_PSNR(GT, SR, border)
% remove border
GT = GT(border+1:end-border, border+1:end-border, :);
SR = SR(border+1:end-border, border+1:end-border, :);
% calculate PNSR (assume in [0,255])
error = GT(:) - SR(:);
mse = mean(error.^2);
res = 10 * log10(255^2/mse);
end

function res = calculate_SSIM(GT, SR, border)
GT = GT(border+1:end-border, border+1:end-border, :);
SR = SR(border+1:end-border, border+1:end-border, :);
% calculate SSIM
mssim = zeros(1, size(SR, 3));
for i = 1:size(SR,3)
    [mssim(i), ~] = ssim_index(GT(:,:,i), SR(:,:,i));
end
res = mean(mssim);
end

function [mssim, ssim_map] = ssim_index(img1, img2, K, window, L)

%========================================================================
%SSIM Index, Version 1.0
%Copyright(c) 2003 Zhou Wang
%All Rights Reserved.
%
%The author is with Howard Hughes Medical Institute, and Laboratory
%for Computational Vision at Center for Neural Science and Courant
%Institute of Mathematical Sciences, New York University.
%
%----------------------------------------------------------------------
%Permission to use, copy, or modify this software and its documentation
%for educational and research purposes only and without fee is hereby
%granted, provided that this copyright notice and the original authors'
%names appear on all copies and supporting documentation. This program
%shall not be used, rewritten, or adapted as the basis of a commercial
%software or hardware product without first obtaining permission of the
%authors. The authors make no representations about the suitability of
%this software for any purpose. It is provided "as is" without express
%or implied warranty.
%----------------------------------------------------------------------
%
%This is an implementation of the algorithm for calculating the
%Structural SIMilarity (SSIM) index between two images. Please refer
%to the following paper:
%
%Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image
%quality assessment: From error measurement to structural similarity"
%IEEE Transactios on Image Processing, vol. 13, no. 1, Jan. 2004.
%
%Kindly report any suggestions or corrections to zhouwang@ieee.org
%
%----------------------------------------------------------------------
%
%Input : (1) img1: the first image being compared
%        (2) img2: the second image being compared
%        (3) K: constants in the SSIM index formula (see the above
%            reference). defualt value: K = [0.01 0.03]
%        (4) window: local window for statistics (see the above
%            reference). default widnow is Gaussian given by
%            window = fspecial('gaussian', 11, 1.5);
%        (5) L: dynamic range of the images. default: L = 255
%
%Output: (1) mssim: the mean SSIM index value between 2 images.
%            If one of the images being compared is regarded as
%            perfect quality, then mssim can be considered as the
%            quality measure of the other image.
%            If img1 = img2, then mssim = 1.
%        (2) ssim_map: the SSIM index map of the test image. The map
%            has a smaller size than the input images. The actual size:
%            size(img1) - size(window) + 1.
%
%Default Usage:
%   Given 2 test images img1 and img2, whose dynamic range is 0-255
%
%   [mssim ssim_map] = ssim_index(img1, img2);
%
%Advanced Usage:
%   User defined parameters. For example
%
%   K = [0.05 0.05];
%   window = ones(8);
%   L = 100;
%   [mssim ssim_map] = ssim_index(img1, img2, K, window, L);
%
%See the results:
%
%   mssim                        %Gives the mssim value
%   imshow(max(0, ssim_map).^4)  %Shows the SSIM index map
%
%========================================================================


if (nargin < 2 || nargin > 5)
    ssim_index = -Inf;
    ssim_map = -Inf;
    return;
end

if (size(img1) ~= size(img2))
    ssim_index = -Inf;
    ssim_map = -Inf;
    return;
end

[M, N] = size(img1);

if (nargin == 2)
    if ((M < 11) || (N < 11))
        ssim_index = -Inf;
        ssim_map = -Inf;
        return
    end
    window = fspecial('gaussian', 11, 1.5);	%
    K(1) = 0.01;								      % default settings
    K(2) = 0.03;								      %
    L = 255;                                  %
end

if (nargin == 3)
    if ((M < 11) || (N < 11))
        ssim_index = -Inf;
        ssim_map = -Inf;
        return
    end
    window = fspecial('gaussian', 11, 1.5);
    L = 255;
    if (length(K) == 2)
        if (K(1) < 0 || K(2) < 0)
            ssim_index = -Inf;
            ssim_map = -Inf;
            return;
        end
    else
        ssim_index = -Inf;
        ssim_map = -Inf;
        return;
    end
end

if (nargin == 4)
    [H, W] = size(window);
    if ((H*W) < 4 || (H > M) || (W > N))
        ssim_index = -Inf;
        ssim_map = -Inf;
        return
    end
    L = 255;
    if (length(K) == 2)
        if (K(1) < 0 || K(2) < 0)
            ssim_index = -Inf;
            ssim_map = -Inf;
            return;
        end
    else
        ssim_index = -Inf;
        ssim_map = -Inf;
        return;
    end
end

if (nargin == 5)
    [H, W] = size(window);
    if ((H*W) < 4 || (H > M) || (W > N))
        ssim_index = -Inf;
        ssim_map = -Inf;
        return
    end
    if (length(K) == 2)
        if (K(1) < 0 || K(2) < 0)
            ssim_index = -Inf;
            ssim_map = -Inf;
            return;
        end
    else
        ssim_index = -Inf;
        ssim_map = -Inf;
        return;
    end
end

C1 = (K(1)*L)^2;
C2 = (K(2)*L)^2;
window = window/sum(sum(window));
img1 = double(img1);
img2 = double(img2);

mu1   = filter2(window, img1, 'valid');
mu2   = filter2(window, img2, 'valid');
mu1_sq = mu1.*mu1;
mu2_sq = mu2.*mu2;
mu1_mu2 = mu1.*mu2;
sigma1_sq = filter2(window, img1.*img1, 'valid') - mu1_sq;
sigma2_sq = filter2(window, img2.*img2, 'valid') - mu2_sq;
sigma12 = filter2(window, img1.*img2, 'valid') - mu1_mu2;

if (C1 > 0 && C2 > 0)
    ssim_map = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))./((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2));
else
    numerator1 = 2*mu1_mu2 + C1;
    numerator2 = 2*sigma12 + C2;
    denominator1 = mu1_sq + mu2_sq + C1;
    denominator2 = sigma1_sq + sigma2_sq + C2;
    ssim_map = ones(size(mu1));
    index = (denominator1.*denominator2 > 0);
    ssim_map(index) = (numerator1(index).*numerator2(index))./(denominator1(index).*denominator2(index));
    index = (denominator1 ~= 0) & (denominator2 == 0);
    ssim_map(index) = numerator1(index)./denominator1(index);
end

mssim = mean2(ssim_map);

end
