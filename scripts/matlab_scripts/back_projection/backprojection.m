function [im_h] = backprojection(im_h, im_l, maxIter)

[row_l, col_l,~] = size(im_l);
[row_h, col_h,~] = size(im_h);

p = fspecial('gaussian', 5, 1);
p = p.^2;
p = p./sum(p(:));

im_l = double(im_l);
im_h = double(im_h);

for ii = 1:maxIter
    im_l_s = imresize(im_h, [row_l, col_l], 'bicubic');
    im_diff = im_l - im_l_s;
    im_diff = imresize(im_diff, [row_h, col_h], 'bicubic');
    im_h(:,:,1) = im_h(:,:,1) + conv2(im_diff(:,:,1), p, 'same');
    im_h(:,:,2) = im_h(:,:,2) + conv2(im_diff(:,:,2), p, 'same');
    im_h(:,:,3) = im_h(:,:,3) + conv2(im_diff(:,:,3), p, 'same');
end
