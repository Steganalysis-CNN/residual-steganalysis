function initParams(obj)
% INITPARAM  Initialize the paramers of the DagNN
%   OBJ.INITPARAM() uses the INIT() method of each layer to initialize
%   the corresponding parameters (usually randomly).

% Copyright (C) 2015 Karel Lenc and Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

KV = 1/12*[-1 2 -2 2 -1;
    2 -6 8 -6 2;
    -2 8 -12 8 -2;
    2 -6 8 -6 2;
    -1 2 -2 2 -1];

for l = 1:numel(obj.layers)
    p = obj.getParamIndex(obj.layers(l).params) ;
    params = obj.layers(l).block.initParams() ;
    switch obj.device
        case 'cpu'
            params = cellfun(@gather, params, 'UniformOutput', false) ;
        case 'gpu'
            params = cellfun(@gpuArray, params, 'UniformOutput', false) ;
    end
    if l == 1
         filterMat = single(zeros(5,5,1,3));
         filterMat(:,:,1,1) = single(KV);
         filterMat(:,:,1,2) = single(KV);
         filterMat(:,:,1,3) = single(KV);
         obj.params(p).value = single(filterMat);
    else
        [obj.params(p).value] = deal(params{:}) ;
    end
end
