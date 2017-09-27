function inputs = getDagNNBatch(opts, useGpu, imdb, batch)
% -------------------------------------------------------------------------
% images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
% isVal = ~isempty(batch) && imdb.images.set(batch(1)) ~= 1 ;
% 
% if ~isVal
%     % training
%     im = cnn_get_batch(images, opts, ...
%         'prefetch', nargout == 0) ;
% else
%     % validation: disable data augmentation
%     im = cnn_get_batch(images, opts, ...
%         'prefetch', nargout == 0, ...
%         'transformation', 'none') ;
% end
% 
% if nargout > 0
%     if useGpu
%         im = gpuArray(im) ;
%     end
%     labels = imdb.images.label(batch) ;
%     inputs = {'data', im, 'label', labels} ;
% end

% step up parameters 
opts.imageSize = [256, 256] ;
opts.border = [0, 0] ; 
opts.keepAspect = true ;
opts.numAugments = 1 ;
opts.transformation = 'none'; 
opts.affine = false;
opts.averageImage = [] ;
opts.rgbVariance = zeros(0,3,'single') ;
opts.interpolation = 'bilinear' ;
opts.numThreads = 1 ;
opts.prefetch = false ;
%%opts = vl_argparse(opts, varargin);

% KV = 1/12*[-1 2 -2 2 -1;
%             2 -6 8 -6 2;
%           -2 8 -12 8 -2;
%             2 -6 8 -6 2;
%           -1 2 -2 2 -1];
% 
%  sample_path = strcat(imdb.coverDir, num2str(batch(1)), '.pgm');
%  cover = imread(sample_path);
%  f_cover = filter2(KV, single(cover));
%  im = single(zeros(size(f_cover,1),size(f_cover,2),3,2*length(batch)));
%  
% for i = 1 : length(batch)
%    cover_path = strcat(imdb.coverDir, num2str(batch(i)), '.pgm'); 
%    stego_path = strcat(imdb.stegoDir, num2str(batch(i)), '.pgm');
%    cover = imread(cover_path);
%    stego = imread(stego_path);
%    f_cover = filter2(KV, double(cover));
%    f_stego = filter2(KV, double(stego));
%    
%    im(:, :, 1, 2*i-1) = single(f_cover);
%    im(:, :, 2, 2*i-1) = single(f_cover);
%    im(:, :, 3, 2*i-1) = single(f_cover);
%    
%    im(:, :, 1, 2*i) = single(f_stego);
%    im(:, :, 2, 2*i) = single(f_stego);
%    im(:, :, 3, 2*i) = single(f_stego);
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The input is original image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 sample_path = strcat(imdb.coverDir, num2str(batch(1)), '.pgm');
 cover = imread(sample_path);
 im = single(zeros(size(cover,1),size(cover,2),1,2*length(batch)));
 
for i = 1 : length(batch)
   cover_path = strcat(imdb.coverDir, num2str(batch(i)), '.pgm'); 
   stego_path = strcat(imdb.stegoDir, num2str(batch(i)), '.pgm');
   cover = imread(cover_path);
   stego = imread(stego_path);
   
   im(:, :, 1, 2*i-1) = single(cover);
   im(:, :, 1, 2*i) = single(stego);
end
 
labels = ones(1,2*length(batch)) + (sign((-1).^(1:2*length(batch)))+1)/2;

if nargout > 0
    if useGpu
        im = gpuArray(im) ;
    end
    inputs = {'data', im, 'label', labels} ;
end

end