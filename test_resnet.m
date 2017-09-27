clear all;  close all;
addpath(genpath('dependencies/'));

setup; % set up the date dependencies
opts.imdb       = [];
opts.networkType = 'resnet' ;
 opts.expDir = 'model';

%other parameter settings
opts.batchNormalization = true ;
opts.nClasses = 2;
opts.batchSize = 10;
opts.numAugments = 1 ;
opts.numEpochs = 200;
opts.bn = true;
opts.whitenData = true;
opts.contrastNormalization = true;
opts.meanType = 'image'; % 'pixel' | 'image'
opts.gpus = []; 
opts.checkpointFn = [];

% initiate the resnet
n = 20;
net = res_init(n, 'nClasses', opts.nClasses,...
                        'batchNormalization', opts.batchNormalization, ...
                        'networkType', opts.networkType) ;
                    
imdb = cnn_steganalysis_setup_data();

% -------------------------------------------------------------------------
%                                                                     Learn
% -------------------------------------------------------------------------
trainfn = @cnn_train_dag_check;

[net, info] = trainfn(net, imdb, getBatchFn(opts, net.meta), ...
  'expDir', opts.expDir, ...
  net.meta.trainOpts, ...
  'gpus', opts.gpus, ...
  'batchSize',opts.batchSize,...
  'numEpochs',opts.numEpochs,...
  'val', find(imdb.images.set == 2), ...
  'derOutputs', {'loss', 1}, ...
  'checkpointFn', opts.checkpointFn) ;

