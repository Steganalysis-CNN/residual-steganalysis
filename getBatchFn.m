function fn = getBatchFn(opts, meta)
% -------------------------------------------------------------------------
useGpu = numel(opts.gpus) > 0 ;

%bopts.numThreads = opts.numFetchThreads ;
bopts.imageSize = meta.normalization.imageSize ;
bopts.border = meta.normalization.border ;
bopts.averageImage = meta.normalization.averageImage ;
bopts.rgbVariance = meta.augmentation.rgbVariance ;
bopts.transformation = meta.augmentation.transformation ;
bopts.numAugments = opts.numAugments ;

switch lower(opts.networkType)
    case 'simplenn'
        fn = @(x,y) getSimpleNNBatch(bopts,x,y) ;
    case {'dagnn', 'resnet'}
        fn = @(x,y) getDagNNBatch(bopts,useGpu,x,y) ;
end
end